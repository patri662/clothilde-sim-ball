import numpy as np
from scipy.spatial import cKDTree

def createMesh(interval, npx, npy, f1, f2, f3):
    #se crea a partir de una parametrizacion de la forma (f1(x,y),f2(x,y),f3(x,y))
    #donde (x,y) estan en el rectangulo [ax,bx]x[ay,by] and interval = [ax bx ay by]
    # Allocate space for the nodal coordinates matrix
    X = np.zeros((npx*npy, 3))
    xs = np.linspace(interval[0], interval[1], npx).reshape(-1, 1)
    unos = np.ones((npx, 1))
    # Nodes' coordinates
    yys = np.linspace(interval[2], interval[3], npy)
    for i in range(npy):
        ys = yys[i] * unos
        posi = np.arange((i * npx), ((i + 1) * npx))
        X[posi, :] = np.column_stack((f1(xs, ys), f2(xs, ys), f3(xs, ys)))
    # Elements (quadrilaterals)
    nx = npx - 1
    ny = npy - 1
    T = np.zeros((nx * ny, 4), dtype=int)
    for a in range(1, ny + 1):
        for b in range(1, nx + 1):
            ielem = (a - 1) * nx + b - 1
            inode = (a - 1) * npx + b - 1
            T[ielem, :] = [inode, inode + 1, inode + npx + 1, inode + npx]
    return X, T

def createRectangularMesh(a,b,na,nb,h = 0.5):
    #coordinate function for a flat cloth
    def f1(x, y):
        return x
    def f2(x, y):
        return y
    def f3(x, y):
        return h*(0.25-x**2) #avoid singular case
    #rectangle; a and b are the sides of the rectangle and na nb the number of nodes
    rect = [-a/2, a/2, -b/2, b/2]
    #create the mesh
    X, T = createMesh(rect, na, nb, f1, f2, f3)   
    return X, T  

def quad_cylinder_mesh(R, H, h, f=1.0):
    """
    Quad mesh of a (possibly flattened) cylinder.

    Parameters
    ----------
    R : float
        Base radius
    H : float
        Height
    h : float
        Target quad edge length
    alpha : float, optional
        Flattening factor in x-direction (alpha=1 -> circular cylinder,
        alpha<1 -> flattened / elliptical cylinder)

    Returns
    -------
    V : (N, 3) ndarray
        Vertex positions
    F : (M, 4) ndarray
        Quad faces
    """

    n_theta = max(3, int(round(2 * np.pi * R / h)))
    n_z     = max(1, int(round(H / h)))

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z = np.linspace(0.0, H, n_z + 1)

    Theta, Z = np.meshgrid(theta, z, indexing="ij")

    X = f * R * np.cos(Theta)
    Y = R * np.sin(Theta)

    V = np.column_stack((X.ravel(), Z.ravel(), Y.ravel()))

    F = []
    for i in range(n_theta):
        ip = (i + 1) % n_theta
        for j in range(n_z):
            v0 = i  * (n_z + 1) + j
            v1 = ip * (n_z + 1) + j
            v2 = ip * (n_z + 1) + (j + 1)
            v3 = i  * (n_z + 1) + (j + 1)
            F.append([v0, v1, v2, v3])

    return V, np.asarray(F, dtype=np.int64)


def duplicate_node_pairs(X, tol=1e-9):
    """
    Parameters
    ----------
    X : (n, 3) array
        Node positions
    tol : float
        Distance tolerance for considering two nodes identical

    Returns
    -------
    pairs : (r, 2) array of int
        Each row [i, j] means X[i] and X[j] are the same (within tol), with i < j
    """
    X = np.asarray(X)
    tree = cKDTree(X)

    # Get all unordered pairs within tolerance
    pairs = tree.query_pairs(r=tol)

    if not pairs:
        return np.empty((0, 2), dtype=int)

    # Convert set of tuples to sorted array
    pairs = np.array(list(pairs), dtype=int)
    pairs.sort(axis=1)  # ensure (i, j) with i < j
    return pairs


# function to create mesh for the solid
# it builds the positions arrays (X in cloth) that is used in Cloth.addSolid(), with the 'edges' connectivity (T in cloth)
# similar to createMesh() for Cloth: create solid shape as a set of nodes around the 'center' with the 'template' shape
# moving the object in position is just done with 'center' point

def createSolidMesh(center, template, edges=None, faces=None): # faces for rendering
    """
    Parameters
    ----------
    center : (3,) array
        World-space position where solid is placed.
    template : (n_nodes, 3) array
        Local node offsets relative to origin that define shape of solid -> corners of cube, two ends of a bar...
    edges : (n_edges, 2) array, optional
        Connectivity information for the solid's edges. Node index pairs that solid holds at their initial distances from each other.
    faces : (n_faces, 4) array, optional
        Quad connectivity for surface rendering of the solid. If None, no surface is drawn.
    
    Returns
    -------
    positions : (n_nodes, 3) array
        World-space positions of solid's nodes for Cloth.addSolid().
    edges : (n_edges, 2) array
        Connectivity information for the solid's edges. If edges is None, returns empty array.
    faces : (n_faces, 4) array
        Quad connectivity for rendering. If faces is None, returns empty array.

    Example
    -------
    >>> template, edges, faces = cubeTemplate(side=0.4)
    >>> positions, edges, faces = createSolidMesh(center=[0, 0, 2], template=template, edges=edges, faces=faces)
    >>> cloth.solid = cloth.addSolid(positions, rad=0.05, mass=1.0, friction=0.3, edges=edges, faces=faces)
    """

    positions = np.array(template, dtype=float) + np.array(center, dtype=float)
    edges = np.zeros((0, 2), dtype=int) if edges is None else np.array(edges, dtype=int)
    faces = np.zeros((0, 4), dtype=int) if faces is None else np.array(faces, dtype=int)
    
    return positions, edges, faces

# if n_segments = 1, is the simple bar with 2 nodes at the ends
# if n_segments > 1, there are n_segments+1 nodes, evenly spaced along `axis`, centered at the origin
def barTemplate(length=1.0, axis=0, n_segments=1):
    # n_segments+1 nodes, evenly spaced along `axis`, centered at the origin
    coords = np.linspace(-length/2, length/2, n_segments + 1)
    template = np.zeros((n_segments + 1, 3))
    template[:, axis] = coords

    # connect each node only to the next one: a simple chain
    edges = np.column_stack((
        np.arange(n_segments),
        np.arange(1, n_segments + 1)
    ))
    return template, edges

def quadTemplate(side=1.0):
    """4 nodes of a regular quadrilater, edge length `side`.
    Fully connected (6 edges): every pair of vertices in a tetrahedron is an
    edge."""
    base = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],], dtype=float)
    base -= base.mean(axis=0)  # center at the origin
    edge_len = np.linalg.norm(base[0] - base[1])
    template = base * (side / edge_len)  # rescale to the requested edge length
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    return template, edges

def cubeSurfaceTemplate(side=1.0, n=2):
    """
    Cube, centered at the origin, built as 6 flat n x n grids.
 
    n=2 reproduces the original 8-corner-only cube (12 edges, 0 midpoints).
    n=3 adds one midpoint per edge and one center point per face (26 nodes
    total, like a Rubik's cube). Higher n gives a finer grid on each face.
 
    Returns
    -------
    template : (n_nodes, 3) local node offsets, centered at the origin
    edges : (n_edges, 2) unique wireframe edges (for rendering only)
    faces : (n_quads, 4) quad connectivity of the cube surface
    """
    h = side / 2.0
    faces = [
        lambda x, y: (x, y, -h*np.ones_like(x)),   # bottom (z=-h)
        lambda x, y: (x, y,  h*np.ones_like(x)),   # top    (z=+h)
        lambda x, y: (x, -h*np.ones_like(x), y),   # front  (y=-h)
        lambda x, y: (x,  h*np.ones_like(x), y),   # back   (y=+h)
        lambda x, y: (-h*np.ones_like(x), x, y),   # left   (x=-h)
        lambda x, y: ( h*np.ones_like(x), x, y),   # right  (x=+h)
    ]
 
    all_X, all_T, offset = [], [], 0
    for f in faces:
        X, T = createMesh([-h, h, -h, h], n, n,
                           *[lambda x, y, i=i, f=f: f(x, y)[i] for i in range(3)])
        all_X.append(X)
        all_T.append(T + offset)
        offset += X.shape[0]
    X = np.vstack(all_X)
    T = np.vstack(all_T)
 
    # merge coincident nodes shared between adjacent faces (exact match,
    # since every face reuses the same linspace(-h,h,n) endpoints/interior points)
    X_unique, inverse = np.unique(X, axis=0, return_inverse=True)
    T_merged = inverse[T]
 
    # derive wireframe edges from the quad connectivity, for rendering only
    edges = np.vstack([T_merged[:, [0, 1]], T_merged[:, [1, 2]],
                        T_merged[:, [2, 3]], T_merged[:, [3, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
 
    return X_unique, edges, T_merged # T_merged are the faces for rendering, not used for physics