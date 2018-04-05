from .build_graph import build_graph
import numpy as np
from scipy.stats import mannwhitneyu
from .w2vec import Doc2vec
import matplotlib.pyplot as plt
from tqdm import tqdm


def build_doc_graph(doc, w2v, dict_size):
    d2v = Doc2vec(doc, w2v, dict_size)
    sim=d2v.vocab_similarity()
    sim=(sim-np.min(sim))/(np.max(sim)-np.min(sim))
    return build_graph(sim)


def build_doc_sim(doc, w2v, dict_size):
    d2v = Doc2vec(doc, w2v, dict_size)
    sim = d2v.vocab_similarity()
    sim=(sim-np.min(sim))/(np.max(sim)-np.min(sim))
    return sim


def build_doc_semantic_graph(doc, w2v, dict_size):
    d2v = Doc2vec(doc, w2v, dict_size)
    sim = d2v.semantic_similarity()
    sim=(sim-np.min(sim))/(np.max(sim)-np.min(sim))
    return build_graph(sim)


def build_doc_semantic_sim(doc, w2v, dict_size):
    d2v = Doc2vec(doc, w2v, dict_size)
    sim=d2v.semantic_similarity()
    sim=(sim-np.min(sim)+10e-3)/(np.max(sim)-np.min(sim)+10e-3)
    return sim


def matrix_spectrum(mat, center=True, standardize=True):
    if center:
        mat = mat - np.mean(mat)
    if standardize:
        mat = mat / np.std(mat)
    mat *= 1 / np.sqrt(mat.shape[0])
    eigval, eigvec = np.linalg.eig(mat)
    return eigvec, eigval


def GraphSpikedTST_PC(docA, docB, w2v, max_len=100000, n_graph=100, plot_spectrum=False):
    sim_A = build_doc_semantic_sim(docA, w2v, max_len)
    sim_B = build_doc_semantic_sim(docB, w2v, max_len)
    n_eigen_A = sim_A.shape[0]
    n_eigen_B = sim_B.shape[0]
    eigvals_A = np.zeros(n_eigen_A * n_graph)
    eigvals_B = np.zeros(n_eigen_B * n_graph)

    for i in tqdm(range(n_graph)):
        graph_A = build_graph(sim_A)
        graph_B = build_graph(sim_B)
        eigvect_A, eigval_A = matrix_spectrum(graph_A)
        eigvect_B, eigval_B = matrix_spectrum(graph_B)
        eigvals_A[i * n_eigen_A:(i + 1) * n_eigen_A] = eigval_A
        eigvals_B[i * n_eigen_B:(i + 1) * n_eigen_B] = eigval_B

    spike_A = (eigvals_A > 2.05 + np.mean(eigvals_A)) + (eigvals_A < -2.05 + np.mean(eigvals_A))
    spike_B = (eigvals_B > 2.05 + np.mean(eigvals_B)) + (eigvals_B < -2.05 + np.mean(eigvals_B))
    eig_spike_A = eigvals_A[spike_A]
    eig_spike_B = eigvals_B[spike_B]

    if plot_spectrum:
        plt.hist(eigval_A, alpha=0.5, bins=100, density=True)
        plt.hist(eigval_B, alpha=0.5, bins=100, density=True)
        plt.title("Spectrum")
        plt.show()

        plt.hist(eig_spike_A, alpha=0.5, bins=50, density=True)
        plt.hist(eig_spike_B, alpha=0.5, bins=50, density=True)
        plt.title("Spike Spectrum")
        plt.show()

    print("# Spike A:", len(eig_spike_A))
    print("# Spike B:", len(eig_spike_B))

    U_test_spike = mannwhitneyu(eig_spike_A, eig_spike_B, alternative='two-sided')[1]
    U_test = mannwhitneyu(eigvals_A, eigvals_B, alternative='two-sided')[1]

    return U_test, U_test_spike, eigvals_A, eigvals_B


def GraphSpikedTST_Cov(docA, docB, w2v, max_len=100000, random=1, n_graph=100, plot_spectrum=False):
    n_words = 1024

    eigvals_A = np.zeros(n_words * n_graph)
    eigvals_B = np.zeros(n_words * n_graph)
    d2vA = Doc2vec(docA, w2v, max_len)
    d2vB = Doc2vec(docB, w2v, max_len)

    for i in tqdm(range(n_graph)):
        sim_A = d2vA.vocab_similarity(size=n_words)
        sim_B = d2vB.vocab_similarity(size=n_words)

        graph_A = build_graph(sim_A)
        graph_B = build_graph(sim_B)
        eigvect_A, eigval_A = matrix_spectrum(graph_A)
        eigvect_B, eigval_B = matrix_spectrum(graph_B)
        eigvals_A[i * n_words:(i + 1) * n_words] = eigval_A
        eigvals_B[i * n_words:(i + 1) * n_words] = eigval_B

    spike_A = (eigvals_A > 2.05 + np.mean(eigvals_A)) + (eigvals_A < -2.05 + np.mean(eigvals_A))
    spike_B = (eigvals_B > 2.05 + np.mean(eigvals_B)) + (eigvals_B < -2.05 + np.mean(eigvals_B))
    eig_spike_A = eigvals_A[spike_A]
    eig_spike_B = eigvals_B[spike_B]

    if plot_spectrum:
        plt.hist(eigval_A, alpha=0.5, bins=100, density=True)
        plt.hist(eigval_B, alpha=0.5, bins=100, density=True)
        plt.title("Spectrum")
        plt.show()

        plt.hist(eig_spike_A, alpha=0.5, bins=50, density=True)
        plt.hist(eig_spike_B, alpha=0.5, bins=50, density=True)
        plt.title("Spike Spectrum")
        plt.show()

    print("# Spike A:", len(eig_spike_A))
    print("# Spike B:", len(eig_spike_B))

    U_test_spike = mannwhitneyu(eig_spike_A, eig_spike_B, alternative='two-sided')[1]
    U_test = mannwhitneyu(eigvals_A, eigvals_B, alternative='two-sided')[1]

    return U_test, U_test_spike, eigvals_A, eigvals_B