
import numpy as np
import cvxopt.base
import cvxopt.solvers
import networkx

def parse_graph(G, complement=False):

    if type(G).__module__+'.'+type(G).__name__ == 'networkx.classes.graph.Graph':
        import networkx
        G = networkx.convert_node_labels_to_integers(G)
        nv = len(G)
        edges = [ (i,j) for (i,j) in G.edges() if i != j ]
        c_edges = [ (i,j) for (i,j) in networkx.complement(G).edges() if i != j ]
    else:
        if type(G).__module__+'.'+type(G).__name__ == 'sage.graphs.graph.Graph':
            G = G.adjacency_matrix().numpy()

        G = np.array(G)

        nv = G.shape[0]
        assert len(G.shape) == 2 and G.shape[1] == nv
        assert np.all(G == G.T)

        edges   = [ (j,i) for i in range(nv) for j in range(i) if G[i,j] ]
        c_edges = [ (j,i) for i in range(nv) for j in range(i) if not G[i,j] ]

    for (i,j) in edges:
        assert i < j
    for (i,j) in c_edges:
        assert i < j

    if complement:
        (edges, c_edges) = (c_edges, edges)

    return (nv, edges, c_edges)

def lovasz_theta(G, long_return=False, complement=False):


    (nv, edges, _) = parse_graph(G, complement)
    ne = len(edges)

    if nv == 1:
        return 1.0

    c = cvxopt.matrix([0.0]*ne + [1.0])
    G1 = cvxopt.spmatrix(0, [], [], (nv*nv, ne+1))
    for (k, (i, j)) in enumerate(edges):
        G1[i*nv+j, k] = 1
        G1[j*nv+i, k] = 1
    for i in range(nv):
        G1[i*nv+i, ne] = 1

    G1 = -G1
    h1 = -cvxopt.matrix(1.0, (nv, nv))

    sol = cvxopt.solvers.sdp(c, Gs=[G1], hs=[h1])

    if long_return:
        theta = sol['x'][ne]
        Z = np.array(sol['ss'][0])
        B = np.array(sol['zs'][0])
        return { 'theta': theta, 'Z': Z, 'B': B }
    else:
        return sol['x'][ne]





theta = lovasz_theta

def thbar(G, long_return=False):
    return lovasz_theta(G, long_return, complement=True)

if __name__ == "__main__":
  G = networkx.cycle_graph(5)
  print(lovasz_theta(G))
