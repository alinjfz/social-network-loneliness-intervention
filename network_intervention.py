import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.stats as st
from community import best_partition
import csv
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import seaborn as sns
from pathlib import Path
import multiprocessing as mp

file_p = './facebook_combined.txt'
DATA_FILE = Path(file_p)  # adjust if needed

SEED              = 42
Q                 = 0.20   
CLUST_FLOOR       = 0.1
N_ROUNDS          = 5
MAX_EDGES_ROUND   = 200
SI_STEPS          = 3
N_SEEDS           = 30
K_SUGG_PER_NODE   = 5
LAYOUT            = 0
random.seed(SEED)
np.random.seed(SEED)

def load_graph(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return nx.read_edgelist(path, nodetype=int)

def metrics(G):
    return {
        "deg":   dict(G.degree()),
        "clust": nx.clustering(G),
        "aspl":  nx.average_shortest_path_length(G),
    }

def left_behind(G, M):
    deg_thr   = np.quantile(list(M["deg"].values()), Q)
    clust_thr = max(CLUST_FLOOR, np.quantile(list(M["clust"].values()), Q))
    return [n for n in G if M["deg"][n] < deg_thr and M["clust"][n] < clust_thr]

def jaccard(G, u, v):
    Nu, Nv = set(G.neighbors(u)), set(G.neighbors(v))
    return 0.0 if not (Nu or Nv) else len(Nu & Nv) / len(Nu | Nv)

def top_hub(G, exclude):
    return max((n for n in G if n not in exclude), key=lambda n: G.degree(n))

def suggest_many(G, community, u, k = K_SUGG_PER_NODE):
    res = []
    # Suggest from the same community
    cand_in = [v for v in G
               if community[v] == community[u] and v != u and not G.has_edge(u, v)]
    scored  = sorted(((v, jaccard(G, u, v)) for v in cand_in),
                     key=lambda t: t[1], reverse=True)
    for v, s in scored[:k]:
        res.append((v, s))
    if len(res) >= k:
        return res[:k]
    # Suggest from other communities
    cand_out = [v for v in G if v != u and community[v] != community[u]
                and not G.has_edge(u, v)]
    scored   = sorted(((v, jaccard(G, u, v)) for v in cand_out),
                      key=lambda t: t[1], reverse=True)
    for v, s in scored:
        if len(res) >= k:
            break
        res.append((v, s))
    if len(res) >= k:
        return res[:k]
    
def plot_network(G, title,community,left_behind,
                 new_edges,edges_added,left_behind_count):
    left_behind = set(left_behind or [])
    plt.figure(figsize=(10, 8))
    comm_ids   = sorted(set(community.values()))
    palette    = plt.cm.tab20(np.linspace(0, 1, len(comm_ids)))
    color_map  = {cid: palette[i] for i, cid in enumerate(comm_ids)}
    node_colors = [
        "red" if n in left_behind else color_map[community[n]]
        for n in G.nodes
    ]
    nx.draw(G, pos=LAYOUT,
            node_color=node_colors,
            node_size=25,
            edge_color="lightgray",
            linewidths=.2)
    if new_edges:
        nx.draw_networkx_edges(G, pos=LAYOUT,
                               edgelist=new_edges,
                               edge_color="red",
                               width=2)    
    if (edges_added > 0 and left_behind_count > 0):
        plt.text(0.95, 0.05, f"Edges added: {edges_added}", ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color="black")
        plt.text(0.95, 0.02, f"Left-behind nodes: {left_behind_count}", ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color="black")
    plt.title(title)
    plt.axis("off")
    plt.show()

def plot_improvements(aspl_values, clustering_values, edges_added_values, left_behind_values, save_dir=None, prefix="figE"):
    iterations = list(range(1, len(aspl_values) + 1))
    plt.figure(figsize=(12, 5))

    # ASPL
    plt.subplot(1, 2, 1)
    plt.plot(iterations, aspl_values, marker='o', color='steelblue')
    plt.title('ASPL Improvement')
    plt.xlabel('Iteration')
    plt.ylabel('ASPL')
    plt.grid(True)
    for i in range(len(aspl_values)):
        plt.text(iterations[i], aspl_values[i], f"{aspl_values[i]:.4f}", fontsize=9, color='steelblue')

    # Clustering
    plt.subplot(1, 2, 2)
    plt.plot(iterations, clustering_values, marker='o', color='orange')
    plt.title('Average Clustering Improvement')
    plt.xlabel('Iteration')
    plt.ylabel('Avg Clustering')
    plt.grid(True)
    for i in range(len(clustering_values)):
        plt.text(iterations[i], clustering_values[i], f"{clustering_values[i]:.4f}", fontsize=9, color='orange')

    if save_dir:
        plt.savefig(Path(save_dir) / f"{prefix}_lineplots.png")
    plt.close()

    # Bar plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(iterations, edges_added_values, color='blue', alpha=0.7)
    ax1.set_title("Edges Added at Each Iteration")
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Edges Added')
    ax1.grid(True)

    ax2.bar(iterations, left_behind_values, color='orange', alpha=0.7)
    ax2.set_title("Left-Behind Nodes at Each Iteration")
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Number of Left-Behind Nodes')
    ax2.grid(True)

    if save_dir:
        fig.savefig(Path(save_dir) / f"{prefix}_barplots.png")
    plt.close()

def cohen_d_paired(x, y):
    diff = np.array(x) - np.array(y)
    return diff.mean() / diff.std(ddof=1)

def one_run(r):
    seed = 42 + r
    rng  = random.Random(seed)
    np.random.seed(seed)
    G0 = load_graph(DATA_FILE)

    # Targeted
    G_t   = G0.copy()
    C_t0  = np.mean(list(nx.clustering(G_t).values()))
    hist_t = improve(G_t, strategy="targeted", rng=rng)
    C_t1  = np.mean(list(nx.clustering(G_t).values()))
    delta_t = C_t1 - C_t0

    # Random
    G_r   = G0.copy()
    C_r0  = np.mean(list(nx.clustering(G_r).values()))
    hist_r = improve(G_r, strategy="random", rng=rng)
    C_r1  = np.mean(list(nx.clustering(G_r).values()))
    delta_r = C_r1 - C_r0

    return (delta_t,
            delta_r,
            hist_t["clust_L"],   # per-round C(L) targeted
            hist_r["clust_L"],   # per-round C(L) random
            hist_t["edges_added"],
            hist_r["edges_added"],
            hist_t,
            hist_r)

def experiment(R=50, out_dir = "figures", save_csv = True):
    out_dir = Path(out_dir) 
    out_dir.mkdir(exist_ok=True, parents=True)
    G0 = load_graph(DATA_FILE)
    global LAYOUT
    LAYOUT = nx.spring_layout(G0, seed=42)
    print(f"Graph loaded: V={G0.number_of_nodes():,}, E={G0.number_of_edges():,}")

    M0      = metrics(G0)
    L0      = left_behind(G0, M0)
    base_cl = [M0["clust"][n] for n in L0]


    n_jobs = mp.cpu_count()
    with mp.Pool(processes=n_jobs) as pool:
        results = list(pool.imap_unordered(one_run, range(R)))
    delta_target, delta_random = [], []
    rounds_target, rounds_random = [], []
    edges_target,  edges_random  = [], []
    history_t,  history_r  = [], []

    for dt, dr, rt, rr, et, er, hist_t, hist_r in results:
        delta_target.append(dt)
        delta_random.append(dr)
        rounds_target.append(rt)
        rounds_random.append(rr)
        history_t.append(hist_t)
        history_r.append(hist_r)

    t_stat, p_val = st.ttest_rel(delta_target, delta_random)
    d_val         = cohen_d_paired(delta_target, delta_random)
    print(f"delta C targeted  mean={np.mean(delta_target):.4f}")
    print(f"delta C random    mean={np.mean(delta_random):.4f}")
    print(f"paired t-test t={t_stat:.2f},  p={p_val:.4g}")

    if save_csv:
        with open(Path(out_dir) / "table1_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "sd"])
            w.writerow(["delta_target", np.mean(delta_target), np.std(delta_target, ddof=1)])
            w.writerow(["delta_random", np.mean(delta_random), np.std(delta_random, ddof=1)])
            w.writerow(["t_stat", t_stat, ""])
            w.writerow(["p_val",  p_val,  ""])
            w.writerow(["cohen_d", d_val, ""])

    COL_TGT, COL_RND = sns.color_palette("tab10")[:2]

    fig, ax = plt.subplots(constrained_layout=True)
    ax.hist(base_cl, bins=30, color=COL_RND, edgecolor="black")
    ax.set_xlabel("Local clustering (peripheral nodes)")
    ax.set_ylabel("Frequency")
    ax.set_title(r"Baseline clustering of peripheral set $L$")
    ax.text(0.98, 0.94, f"N = {len(base_cl)}", ha="right", va="top",
            transform=ax.transAxes, fontsize=10)

    axins = inset_axes(ax, width="40%", height="40%", loc="upper right",
                   bbox_to_anchor=(0.55, 0.55, 0.4, 0.4),  # x0, y0, width, height
                   bbox_transform=ax.transAxes)

    axins.hist([c for c in base_cl if c > .25], bins=10,
            color=COL_RND, edgecolor="black")
    axins.set_xlim(.25, .45)
    axins.set_xticks([.3, .35, .4, .45])
    axins.set_yticks([])           
    axins.set_title("Right tail", fontsize=9)
    out_dir = Path(out_dir) 
    fig.savefig(out_dir / "figA_baseline_hist.png")

    plt.close(fig)

    fig, ax = plt.subplots(constrained_layout=True)
    box = ax.boxplot([delta_target, delta_random],
                    positions=[1, 2],
                    patch_artist=True,
                    showmeans=True,
                    widths=.6)
    for patch, col in zip(box['boxes'], [COL_TGT, COL_RND]):
        patch.set_facecolor(col)
    for median in box['medians']:
        median.set_color("black")

    ax.set_xticks([1, 2]); ax.set_xticklabels(["Targeted", "Random"])
    ax.set_ylabel(r"$\Delta$ local clustering of $L$")
    ax.set_title("Figure B - Paired comparison of clustering gains")

    y_star = max(max(delta_target), max(delta_random)) + 0.001
    ax.plot([1, 2], [y_star, y_star], color="black", lw=1)
    ax.text(1.5, y_star + 0.0003, "***" if p_val < .001 else "**",
            ha="center", va="bottom", fontsize=14)

    fig.savefig(out_dir / "figB_boxplot_deltaC.png")
    plt.close(fig)

    rounds_t = np.array([v + [np.nan]*(N_ROUNDS-len(v)) for v in rounds_target])
    rounds_r = np.array([v + [np.nan]*(N_ROUNDS-len(v)) for v in rounds_random])
    x_axis   = np.arange(1, N_ROUNDS+1)

    mean_t, se_t = np.nanmean(rounds_t, 0), np.nanstd(rounds_t, 0, ddof=1)/np.sqrt(len(rounds_t))
    mean_r, se_r = np.nanmean(rounds_r, 0), np.nanstd(rounds_r, 0, ddof=1)/np.sqrt(len(rounds_r))

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(x_axis, mean_t, marker="o", color=COL_TGT, label="Targeted")
    ax.fill_between(x_axis, mean_t-se_t, mean_t+se_t, color=COL_TGT, alpha=.25)
    ax.plot(x_axis, mean_r, marker="s", color=COL_RND, label="Random")
    ax.fill_between(x_axis, mean_r-se_r, mean_r+se_r, color=COL_RND, alpha=.25)

    ax.set_xlabel("Round")
    ax.set_ylabel("Mean clustering of $L$")
    ax.set_title("Figure C - Clustering trajectory across rounds")
    ax.legend(loc="upper left")
    
    fig.savefig(out_dir / "figC_trajectory.png")
    plt.close(fig)

    edges_all = []
    gain_all  = []
    colors_all = []
    markers    = []
    if edges_target and edges_random:
        edges_all  = np.concatenate(edges_target + edges_random)
        gain_all   = np.concatenate(rounds_target + rounds_random)
        colors_all = ([COL_TGT]*len(np.concatenate(edges_target)) +
                    [COL_RND]*len(np.concatenate(edges_random)))
        markers    = (["o"]*len(np.concatenate(edges_target)) +
                    ["s"]*len(np.concatenate(edges_random)))

    fig, ax = plt.subplots()
    for x, y, c, m in zip(edges_all, gain_all, colors_all, markers):
        ax.scatter(x, y, color=c, alpha=.7, marker=m)

    if (edges_target and edges_random):
        for xs, ys, col in [(np.concatenate(edges_target), np.concatenate(rounds_target), COL_TGT),
                            (np.concatenate(edges_random), np.concatenate(rounds_random), COL_RND)]:
            m, b = np.polyfit(xs, ys, 1)
            xfit = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(xfit, m*xfit + b, color=col, lw=1)

    ax.set_xlabel("Edges added in round")
    ax.set_ylabel("Clustering of $L$ after that round")
    ax.set_title("Figure D - Edges added vs achieved clustering")

    fig.savefig(out_dir / "figD_edges_vs_gain.png")
    plt.close(fig)
    avg_aspl = np.nanmean([h["aspl"] for h in history_t], axis=0)
    avg_clust = np.nanmean([h["clust_L"] for h in history_t], axis=0)
    avg_edges = np.nanmean([h["edges_added"] for h in history_t], axis=0)
    avg_left_behind = np.nanmean([h["lb_count"] for h in history_t], axis=0)

    plot_improvements(
        avg_aspl,
        avg_clust,
        avg_edges,
        avg_left_behind,
        save_dir=out_dir,
        prefix="figE"
    )
    return delta_target, delta_random, t_stat, p_val, d_val


def improve(
        G,strategy,        # "targeted"  or  "random"
        rng,n_rounds = N_ROUNDS,
        max_edges_round = MAX_EDGES_ROUND,
        recompute_comm_each_round = True):

    if rng is None:
        rng = random
    hist = dict(
        clust_L=[],   # mean clustering of peripheral nodes
        aspl=[],      # global ASPL
        lb_count=[],  # |L|
        edges_added=[]
    )
    comm = best_partition(G, random_state=rng.randrange(10**6))
    M    = metrics(G)
    L    = left_behind(G, M)

    for itr in range(1, n_rounds + 1):
        if recompute_comm_each_round and itr > 1:
            comm = best_partition(G, random_state=rng.randrange(10**6))

        suggestions: list[tuple[float, int, int]] = []   # (score, u, v)

        if strategy == "targeted":
            for u in L:
                for v, score in suggest_many(G, comm, u, K_SUGG_PER_NODE):
                    suggestions.append((score, u, v))
            # sort high-to-low by score (Jaccard)
            suggestions.sort(reverse=True, key=lambda t: t[0])

        elif strategy == "random":
            cand = [(u, v) for u in L
                           for v in G
                           if comm[v] == comm[u] and v != u and not G.has_edge(u, v)]
            rng.shuffle(cand)
            suggestions = [(0.0, u, v) for (u, v) in cand]

        else:
            raise ValueError("strategy must be 'targeted' or 'random'")

        used: set[int] = set()
        new_edges: list[tuple[int, int]] = []
        for _, u, v in suggestions:
            if len(new_edges) >= max_edges_round:
                break
            if u in used or v in used or G.has_edge(u, v):
                continue
            new_edges.append((u, v))
            used.update([u, v])

        if not new_edges:
            # nothing left to add
            break

        G.add_edges_from(new_edges)

        M = metrics(G)
        L = left_behind(G, M)

        clust_L_mean = np.mean([M["clust"][n] for n in L]) if L else np.nan
        hist["clust_L"].append(clust_L_mean)
        hist["aspl"].append(M["aspl"])
        hist["lb_count"].append(len(L))
        hist["edges_added"].append(len(new_edges))
        if not L:
            break
    return hist

if __name__ == "__main__":
    print("Start!")
    delta_target, delta_random, t_stat, p_val, d_val = experiment(3)
