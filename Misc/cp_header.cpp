/******** All Required Header Files ********/
#include <bits/stdc++.h>
#include <unordered_map>
#include <unordered_set>

using namespace std;

/******* All Required define Pre-Processors and typedef Constants *******/
typedef long long ll;
typedef long double ld;
typedef pair<int, int> ii;
typedef pair<int, ii> iii;
typedef pair<long, long> pll;
typedef vector<string> vs;
typedef vector<int> vi; // Change it to ll if numbers are large!!!
typedef vector<ii> vii;
typedef vector<vi> vvi;
typedef vector<vii> vvii;
typedef vector<vvi> vvvi;
typedef vector<vvii> vvvii;
typedef set<int> si;
typedef vector<si> vsi;
typedef vector<bool> vb;
typedef vector<vb> vvb;
typedef priority_queue<long> max_heap;
typedef priority_queue<long, vector<long>, greater<long>> min_heap;
typedef map<int, int> mapii;
typedef unordered_map<int, int> umapii;
typedef set<int> seti;
typedef unordered_set<int> useti;
typedef multiset<int> mseti;

#define SCD(t) scanf("\%d", &t)
#define SCLD(t) scanf("\%ld", &t)
#define SCLLD(t) scanf("\%lld", &t)
#define SCC(t) scanf("\%c", &t)
#define SCS(t) scanf("\%s", t)
#define SCF(t) scanf("\%f", &t)
#define SCLF(t) scanf("\%lf", &t)
#define MS(a, b) memset(a, (b), sizeof(a))
#define FOR(i, j, n, step) for (int i = j; i < n; i += step)
#define RFOR(i, j, n, step) for (int i = j; i >= n; i -= step)
#define REP(i, j) FOR(i, 0, j, 1)
#define RREP(i, j) RFOR(i, j, 0, 1)
#define ALL(container) container.begin(), container.end()
#define RALL(container) container.end(), container.begin()
#define FOREACH(it, l) for (auto it = l.begin(); it != l.end(); it++)
#define IN(A, B, C) assert(B <= A && A <= C)
#define LC (node << 1)     // Left child node in binary tree
#define RC (node << 1 | 1) // Right child node in binary tree
#define MP make_pair
#define PB push_back
#define SORT(container) sort(container.begin(), container.end())
#define REVERSE(container) reverse(container.begin(), container.end())
#define DEBUG(a, b) cout << a << ": " << b << endl
#define SANITY cout << "Reached here" << endl
#define PRINT(a) cout << a << "\n"
#define PrintPair(p) cout << p.first << " " << p.second << endl

#define FAST_IO                       \
    ios_base::sync_with_stdio(false); \
    cin.tie(NULL);                    \
    cout.tie(NULL)
#define endl '\n'
#define INF 1e9
#define LONG_INF 1e18
#define EPS 1e-9
#define MOD 1000000007
#define PI acos(-1.0) // important constant; alternative #define PI (2.0 * acos(0.0))

/****** Template of some basic operations *****/
template <typename T>
T gcd(T a, T b) { return b == 0 ? a : gcd(b, a % b); }
template <typename T>
T lcm(T a, T b) { return a * b / gcd(a, b); }
template <typename T>
T power(T base, T exp)
{
    T result = 1;
    while (exp)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }
    return result;
}
ll mulmodn(ll a, ll b, ll n)
{
    ll res = 0;
    while (b)
    {
        if (b & 1)
            res = (res + a) % n;
        a = (a * 2) % n;
        b >>= 1;
    }
    return res;
}

ll powmodn(ll a, ll q, ll n)
{
    ll res = 1;
    while (q)
    {
        if (q & 1)
            res = mulmodn(res, a, n);
        a = mulmodn(a, a, n);
        q >>= 1;
    }
    return res;
}

ll factorial(int n)
{
    ll result = 1;
    for (int i = 2; i <= n; ++i)
    {
        result *= i;
    }
    return result;
}

template <typename T>
T mod_inv(T a, T m) { return power(a, m - 2); }
template <typename T, typename U>
inline void minimum(T &x, U y)
{
    if (y < x)
        x = y;
}
template <typename T, typename U>
inline void maximum(T &x, U y)
{
    if (x < y)
        x = y;
}

/****** Data Structures *********/

///////// Union Find //////////

struct unionfind
{
    vector<int> rank;
    vector<int> parent;

    unionfind(int size)
    {
        rank = vector<int>(size, 0);
        parent = vector<int>(size);
        for (int i = 0; i < size; i++)
            parent[i] = i;
    }

    int find(int x)
    {
        int tmp = x;
        while (x != parent[x])
            x = parent[x];
        while (tmp != x)
        {
            int remember = parent[tmp];
            parent[tmp] = x;
            tmp = remember;
        }
        return x;
    }

    void unite(int p, int q)
    {
        p = find(p);
        q = find(q);
        if (q == p)
            return;
        if (rank[p] < rank[q])
            parent[p] = q;
        else
            parent[q] = p;
        if (rank[p] == rank[q])
            rank[p]++;
    }
};

// int main()
// {
//     /// union find
//     unionfind uf(10);
//     uf.unite(1, 5);
//     uf.unite(2, 5);
//     if (uf.find(1) == uf.find(2))
//         cout << "1 and 2 in the same team" << endl;
//     if (uf.find(1) != uf.find(7))
//         cout << "1 and 7 not in the same team";
//     return 0;
// }

///////// Segment Tree //////////

const int N = 1e5; // limit for array size
int n;             // array size
int t[2 * N];

void build()
{ // build the tree
    for (int i = n - 1; i > 0; --i)
        t[i] = t[i << 1] + t[i << 1 | 1];
}

void modify(int p, int value)
{ // set value at position p
    for (t[p += n] = value; p > 1; p >>= 1)
        t[p >> 1] = t[p] + t[p ^ 1];
}

int query(int l, int r)
{ // sum on interval [l, r)
    int res = 0;
    for (l += n, r += n; l < r; l >>= 1, r >>= 1)
    {
        if (l & 1)
            res += t[l++];
        if (r & 1)
            res += t[--r];
    }
    return res;
}

// int main()
// {
//     n = 15; // array size
//     for (int i = 0; i < n; ++i)
//         t[n + i] = 1; // init array
//     build();
//     modify(0, 1);
//     cout << query(3, 11) << endl;
// }

///////// Fenwick Tree //////////

int BIT[1000] = {0}, a[1000], m;
void update(int x, int val)
{
    for (; x <= m; x += x & -x)
        BIT[x] += val;
}
int query(int x)
{
    int sum = 0;
    for (; x > 0; x -= x & -x)
        sum += BIT[x];
    return sum;
}

// int main()
// {
//     scanf("%d", &n);
//     int i;
//     for (i = 1; i <= n; i++)
//     {
//         scanf("%d", &a[i]);
//         update(i, a[i]);
//     }
//     printf("sum of first 10 elements is %d\n", query(10));
//     printf("sum of all elements in range [2, 7] is %d\n", query(7) - query(2 - 1));
//     return 0;
// }

/****** Graph Theory *********/

/********** Topological Sort **********/

// input: directed graph (g[u] contains the neighbors of u, nodes are named 0,1,...,|V|-1).
// output: is g a DAG (return value), a topological ordering of g (order).
// comment: order is valid only if g is a DAG.
// time: O(V+E).
bool topological_sort(const vvi &g, vi &order)
{
    // compute indegree of all nodes
    vi indegree(g.size(), 0);
    for (int v = 0; v < g.size(); v++)
        for (int u : g[v])
            indegree[u]++;
    // order sources first
    order = vector<int>();
    for (int v = 0; v < g.size(); v++)
        if (indegree[v] == 0)
            order.push_back(v);
    // go over the ordered nodes and remove outgoing edges,
    // add new sources to the ordering
    for (int i = 0; i < order.size(); i++)
        for (int u : g[order[i]])
        {
            indegree[u]--;
            if (indegree[u] == 0)
                order.push_back(u);
        }
    return order.size() == g.size();
}

/********** Strongly Connected Components **********/

const int UNSEEN = -1;
const int SEEN = 1;

void KosarajuDFS(const vvi &g, int u, vi &S, vi &colorMap, int color)
{
    // DFS on digraph g from node u:
    // visit a node only if it is mapped to the color UNSEEN,
    // Mark all visited nodes in the color map using the given color.
    // input: digraph (g), node (v), mapping:node->color (colorMap), color (color).
    // output: DFS post-order (S), node coloring (colorMap).
    colorMap[u] = color;
    for (auto &v : g[u])
        if (colorMap[v] == UNSEEN)
            KosarajuDFS(g, v, S, colorMap, color);
    S.push_back(u);
}

// Compute the number of SCCs and maps nodes to their corresponding SCCs.
// input: directed graph (g[u] contains the neighbors of u, nodes are named 0,1,...,|V|-1).
// output: the number of SCCs (return value), a mapping from node to SCC color (components).
// time: O(V+E).
int findSCC(const vvi &g, vi &components)
{
    // first pass: record the `post-order' of original graph
    vi postOrder, seen;
    seen.assign(g.size(), UNSEEN);
    for (int i = 0; i < g.size(); ++i)
        if (seen[i] == UNSEEN)
            KosarajuDFS(g, i, postOrder, seen, SEEN);
    // second pass: explore the SCCs based on first pass result
    vvi reverse_g(g.size(), vi());
    for (int u = 0; u < g.size(); u++)
        for (int v : g[u])
            reverse_g[v].push_back(u);
    vi dummy;
    components.assign(g.size(), UNSEEN);
    int numSCC = 0;
    for (int i = (int)g.size() - 1; i >= 0; --i)
        if (components[postOrder[i]] == UNSEEN)
            KosarajuDFS(reverse_g, postOrder[i], dummy, components, numSCC++);
    return numSCC;
}

// Computes the SCC graph of a given digraph.
// input: directed graph (g[u] contains the neighbors of u, nodes are named 0,1,...,|V|-1).
// output: strongly connected components graph of g (sccg).
// time: O(V+E).
void findSCCgraph(const vvi &g, vsi &sccg)
{
    vi component;
    int n = findSCC(g, component);
    sccg.assign(n, si());
    for (int u = 0; u < g.size(); u++)
        for (int v : g[u]) // for every edge u->v
            if (component[u] != component[v])
                sccg[component[u]].insert(component[v]);
}

/********** Shortest Paths **********/

// input: non-negatively weighted directed graph (g[u] contains pairs (v,w) such that u->v has weight w, nodes are named 0,1,...,|V|-1), source (s).
// output: distances from s (dist).
// time: O(ElogV).
void Dijkstra(const vvii &g, int s, vi &dist)
{
    dist = vi(g.size(), INF);
    dist[s] = 0;
    priority_queue<ii, vii, greater<ii>> q;
    q.push({0, s});
    while (!q.empty())
    {
        ii front = q.top();
        q.pop();
        int d = front.first, u = front.second;
        if (d > dist[u])
            continue; // We may have found a shorter way to get to u after inserting it to q.
        // In that case, we want to ignore the previous insertion to q.
        for (ii next : g[u])
        {
            int v = next.first, w = next.second;
            if (dist[u] + w < dist[v])
            {
                dist[v] = dist[u] + w;
                q.push({dist[v], v});
            }
        }
    }
}

// input: weighted directed graph (g[u] contains pairs (v,w) such that u->v has weight w, nodes are named 0,1,...,|V|-1), source node (s).
// output: is there a negative cycle in g? (return value), the distances from s (d)
// comment: the values in d are valid only if there is no negative cycle.
// time: O(VE).
bool BellmanFord(const vvii &g, int s, vi &d)
{
    d.assign(g.size(), INF);
    d[s] = 0;
    bool changed = false;
    // V times
    for (int i = 0; i < g.size(); ++i)
    {
        changed = false;
        // go over all edges u->v with weight w
        for (int u = 0; u < g.size(); ++u)
            for (ii e : g[u])
            {
                int v = e.first;
                int w = e.second;
                // relax the edge
                if (d[u] < INF && d[u] + w < d[v])
                {
                    d[v] = d[u] + w;
                    changed = true;
                }
            }
    }
    // there is a negative cycle if there were changes in the last iteration
    return changed;
}

// input: weighted directed graph (g[u] contains pairs (v,w) such that u->v has weight w, nodes are named 0,1,...,|V|-1).
// output: the pairwise distances (d).
// time: O(V^3).
void FloydWarshall(const vvii &g, vvi &d)
{
    // initialize distances according to the graph edges
    d.assign(g.size(), vi(g.size(), INF));
    for (int u = 0; u < g.size(); ++u)
        d[u][u] = 0;
    for (int u = 0; u < g.size(); ++u)
        for (ii e : g[u])
        {
            int v = e.first;
            int w = e.second;
            d[u][v] = min(d[u][v], w);
        }
    // relax distances using the Floyd-Warshall algorithm
    for (int k = 0; k < g.size(); ++k)
        for (int u = 0; u < g.size(); ++u)
            for (int v = 0; v < g.size(); ++v)
                d[u][v] = min(d[u][v], d[u][k] + d[k][v]);
}

/********** Min Spanning Tree **********/

// input: edges v1->v2 of the form (weight,(v1,v2)),
//        number of nodes (n), all nodes are between 0 and n-1.
// output: weight of a minimum spanning tree.
// time: O(ElogV).
int Kruskal(vector<iii> &edges, int n)
{
    sort(edges.begin(), edges.end());
    int mst_cost = 0;
    unionfind components(n);
    for (iii e : edges)
    {
        if (components.find(e.second.first) != components.find(e.second.second))
        {
            mst_cost += e.first;
            components.unite(e.second.first, e.second.second);
        }
    }
    return mst_cost;
}

/********** Max Flow **********/

int augment(vvi &res, int s, int t, const vi &p, int minEdge)
{
    // traverse the path from s to t according to p.
    // change the residuals on this path according to the min edge weight along this path.
    // return the amount of flow that was added.
    if (t == s)
    {
        return minEdge;
    }
    else if (p[t] != -1)
    {
        int f = augment(res, s, p[t], p, min(minEdge, res[p[t]][t]));
        res[p[t]][t] -= f;
        res[t][p[t]] += f;
        return f;
    }
    return 0;
}

// input: number of nodes (n), all nodes are between 0 and n-1,
//        edges v1->v2 of the form (weight,(v1,v2)), source (s) and target (t).
// output: max flow from s to t over the edges.
// time: O(VE^2) and O(EF).
int EdmondsKarp(int n, vector<iii> &edges, int s, int t)
{
    // initialise adjacenty list and residuals adjacency matrix
    vvi res(n, vi(n, 0));
    vvi adj(n);
    for (iii e : edges)
    {
        res[e.second.first][e.second.second] += e.first;
        adj[e.second.first].push_back(e.second.second);
        adj[e.second.second].push_back(e.second.first);
    }
    // while we can add flow
    int addedFlow, maxFlow = 0;
    do
    {
        // save to p the BFS tree from s to t using only edges with residuals
        vi dist(res.size(), INF);
        dist[s] = 0;
        queue<int> q;
        q.push(s);
        vi p(res.size(), -1);
        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            if (u == t)
                break;
            for (int v : adj[u])
                if (res[u][v] > 0 && dist[v] == INF)
                {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                    p[v] = u;
                }
        }
        // add flow on the path between s to t according to p
        addedFlow = augment(res, s, t, p, INF);
        maxFlow += addedFlow;
    } while (addedFlow > 0);
    return maxFlow;
}

void bfs(const vvi &g, int s, vector<int> &d)
{
    queue<int> q;
    q.push(s);
    vector<bool> visible(g.size(), false);
    visible[s] = true;
    d.assign(g.size(), INF);
    d[s] = 0;
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        for (int v : g[u])
            if (!visible[v])
            {
                visible[v] = true;
                d[v] = d[u] + 1;
                q.push(v);
            }
    }
}

void dfs(const vvi &g, int s, vector<int> &d)
{
    stack<int> q;
    q.push(s);
    vector<bool> seen(g.size(), false);
    seen[s] = true;
    d.assign(g.size(), INF);
    d[s] = 0;
    while (!q.empty())
    {
        int u = q.top();
        q.pop();
        for (int v : g[u])
            if (!seen[v])
            {
                seen[v] = true;
                d[v] = d[u] + 1;
                q.push(v);
            }
    }
}

// Checks if the graph is cyclic
bool isCyclic(vector<vector<int>> &graph)
{
    int V = graph.size();
    vector<bool> visited(V, false);
    vector<int> parent(V, -1);

    // Use an explicit stack for DFS
    for (int u = 0; u < V; ++u)
    {
        if (!visited[u])
        {
            stack<int> s;
            s.push(u);

            while (!s.empty())
            {
                int v = s.top();
                s.pop();

                if (!visited[v])
                {
                    visited[v] = true;
                    for (int neighbor : graph[v])
                    {
                        if (!visited[neighbor])
                        {
                            s.push(neighbor);
                            parent[neighbor] = v;
                        }
                        else if (parent[v] != neighbor)
                        {
                            return true; // Cycle detected
                        }
                    }
                }
            }
        }
    }
    return false;
}

/****** Computational Geometry *********/

double DEG_to_RAD(double d) { return d * PI / 180.0; }

double RAD_to_DEG(double r) { return r * 180.0 / PI; }

// struct point_i { int x, y; };    // basic raw form, minimalist mode
struct point_i
{
    int x, y;                // whenever possible, work with point_i
    point_i() { x = y = 0; } // default constructor
    point_i(int _x, int _y) : x(_x), y(_y) {}
}; // user-defined

struct point
{
    double x, y;                                  // only used if more precision is needed
    point() { x = y = 0.0; }                      // default constructor
    point(double _x, double _y) : x(_x), y(_y) {} // user-defined
    bool operator<(point other) const
    {                                // override less than operator
        if (fabs(x - other.x) > EPS) // useful for sorting
            return x < other.x;      // first criteria , by x-coordinate
        return y < other.y;
    } // second criteria, by y-coordinate
    // use EPS (1e-9) when testing equality of two floating points
    bool operator==(point other) const
    {
        return (fabs(x - other.x) < EPS && (fabs(y - other.y) < EPS));
    }
};

double dist(point p1, point p2)
{ // Euclidean distance
    // hypot(dx, dy) returns sqrt(dx * dx + dy * dy)
    return hypot(p1.x - p2.x, p1.y - p2.y);
} // return double

// rotate p by theta degrees CCW w.r.t origin (0, 0)
point rotate(point p, double theta)
{
    double rad = DEG_to_RAD(theta); // multiply theta with PI / 180.0
    return point(p.x * cos(rad) - p.y * sin(rad),
                 p.x * sin(rad) + p.y * cos(rad));
}

struct line
{
    double a, b, c;
}; // a way to represent a line

// the answer is stored in the third parameter (pass by reference)
void pointsToLine(point p1, point p2, line &l)
{
    if (fabs(p1.x - p2.x) < EPS)
    { // vertical line is fine
        l.a = 1.0;
        l.b = 0.0;
        l.c = -p1.x; // default values
    }
    else
    {
        l.a = -(double)(p1.y - p2.y) / (p1.x - p2.x);
        l.b = 1.0; // IMPORTANT: we fix the value of b to 1.0
        l.c = -(double)(l.a * p1.x) - p1.y;
    }
}

// not needed since we will use the more robust form: ax + by + c = 0 (see above)
struct line2
{
    double m, c;
}; // another way to represent a line

int pointsToLine2(point p1, point p2, line2 &l)
{
    if (abs(p1.x - p2.x) < EPS)
    {               // special case: vertical line
        l.m = INF;  // l contains m = INF and c = x_value
        l.c = p1.x; // to denote vertical line x = x_value
        return 0;   // we need this return variable to differentiate result
    }
    else
    {
        l.m = (double)(p1.y - p2.y) / (p1.x - p2.x);
        l.c = p1.y - l.m * p1.x;
        return 1; // l contains m and c of the line equation y = mx + c
    }
}

bool areParallel(line l1, line l2)
{ // check coefficients a & b
    return (fabs(l1.a - l2.a) < EPS) && (fabs(l1.b - l2.b) < EPS);
}

bool areSame(line l1, line l2)
{ // also check coefficient c
    return areParallel(l1, l2) && (fabs(l1.c - l2.c) < EPS);
}

// returns true (+ intersection point) if two lines are intersect
bool areIntersect(line l1, line l2, point &p)
{
    if (areParallel(l1, l2))
        return false; // no intersection
    // solve system of 2 linear algebraic equations with 2 unknowns
    p.x = (l2.b * l1.c - l1.b * l2.c) / (l2.a * l1.b - l1.a * l2.b);
    // special case: test for vertical line to avoid division by zero
    if (fabs(l1.b) > EPS)
        p.y = -(l1.a * p.x + l1.c);
    else
        p.y = -(l2.a * p.x + l2.c);
    return true;
}

struct vec
{
    double x, y; // name: `vec' is different from STL vector
    vec(double _x, double _y) : x(_x), y(_y) {}
};

vec toVec(point a, point b)
{ // convert 2 points to vector a->b
    return vec(b.x - a.x, b.y - a.y);
}

vec scale(vec v, double s)
{ // nonnegative s = [<1 .. 1 .. >1]
    return vec(v.x * s, v.y * s);
} // shorter.same.longer

point translate(point p, vec v)
{ // translate p according to v
    return point(p.x + v.x, p.y + v.y);
}

// convert point and gradient/slope to line
void pointSlopeToLine(point p, double m, line &l)
{
    l.a = -m; // always -m
    l.b = 1;  // always 1
    l.c = -((l.a * p.x) + (l.b * p.y));
} // compute this

void closestPoint(line l, point p, point &ans)
{
    line perpendicular; // perpendicular to l and pass through p
    if (fabs(l.b) < EPS)
    { // special case 1: vertical line
        ans.x = -(l.c);
        ans.y = p.y;
        return;
    }

    if (fabs(l.a) < EPS)
    { // special case 2: horizontal line
        ans.x = p.x;
        ans.y = -(l.c);
        return;
    }

    pointSlopeToLine(p, 1 / l.a, perpendicular); // normal line
    // intersect line l with this perpendicular line
    // the intersection point is the closest point
    areIntersect(l, perpendicular, ans);
}

// returns the reflection of point on a line
void reflectionPoint(line l, point p, point &ans)
{
    point b;
    closestPoint(l, p, b); // similar to distToLine
    vec v = toVec(p, b);   // create a vector
    ans = translate(translate(p, v), v);
} // translate p twice

double dot(vec a, vec b) { return (a.x * b.x + a.y * b.y); }

double norm_sq(vec v) { return v.x * v.x + v.y * v.y; }

// returns the distance from p to the line defined by
// two points a and b (a and b must be different)
// the closest point is stored in the 4th parameter (byref)
double distToLine(point p, point a, point b, point &c)
{
    // formula: c = a + u * ab
    vec ap = toVec(a, p), ab = toVec(a, b);
    double u = dot(ap, ab) / norm_sq(ab);
    c = translate(a, scale(ab, u)); // translate a to c
    return dist(p, c);
} // Euclidean distance between p and c

// returns the distance from p to the line segment ab defined by
// two points a and b (still OK if a == b)
// the closest point is stored in the 4th parameter (byref)
double distToLineSegment(point p, point a, point b, point &c)
{
    vec ap = toVec(a, p), ab = toVec(a, b);
    double u = dot(ap, ab) / norm_sq(ab);
    if (u < 0.0)
    {
        c = point(a.x, a.y); // closer to a
        return dist(p, a);
    } // Euclidean distance between p and a
    if (u > 1.0)
    {
        c = point(b.x, b.y); // closer to b
        return dist(p, b);
    } // Euclidean distance between p and b
    return distToLine(p, a, b, c);
} // run distToLine as above

double angle(point a, point o, point b)
{ // returns angle aob in rad
    vec oa = toVec(o, a), ob = toVec(o, b);
    return acos(dot(oa, ob) / sqrt(norm_sq(oa) * norm_sq(ob)));
}

double cross(vec a, vec b) { return a.x * b.y - a.y * b.x; }

double cross(point o, point a, point b)
{ // returns the cross product of OA and OB vectors
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

//// another variant
// int area2(point p, point q, point r) { // returns 'twice' the area of this triangle A-B-c
//   return p.x * q.y - p.y * q.x +
//          q.x * r.y - q.y * r.x +
//          r.x * p.y - r.y * p.x;
// }

// note: to accept collinear points, we have to change the `> 0' to '>=0'
// returns true if point r is on the left side of line pq (or on the line pq if colinear)
bool ccw(point p, point q, point r)
{
    return cross(toVec(p, q), toVec(p, r)) > 0;
}

// returns true if point r is on the same line as the line pq
bool collinear(point p, point q, point r)
{
    return fabs(cross(toVec(p, q), toVec(p, r))) < EPS;
}

// Andrew's Monotone Chain Algorithm
// returns the convex hull of a set of points
vector<point> convex_hull_monotone(vector<point> Points)
{
    int n = Points.size(), k = 0;
    vector<point> H(2 * n);
    sort(Points.begin(), Points.end());
    for (int i = 0; i < n; i++)
    {
        while (k >= 2 && cross(H[k - 2], H[k - 1], Points[i]) < EPS)
            k--;
        H[k++] = Points[i];
    }
    for (int i = n - 2, t = k + 1; i >= 0; i--)
    {
        while (k >= t && cross(H[k - 2], H[k - 1], Points[i]) < EPS)
            k--;
        H[k++] = Points[i];
    }
    H.resize(k - 1);
    return H;
}

// Incremental Algorithm
// returns the convex hull of a set of points
vector<point> convex_hull_incremental(vector<point> Points)
{
    sort(Points.begin(), Points.end());
    stack<point> stk_up;
    stk_up.push(Points[0]);
    stk_up.push(Points[1]);
    for (int i = 2; i < Points.size(); i++)
    {
        while (stk_up.size() >= 2)
        {
            point p2 = stk_up.top();
            stk_up.pop();
            point p3 = stk_up.top();
            if (ccw(p3, p2, Points[i]))
            {
                stk_up.push(p2);
                break;
            }
        }
        stk_up.push(Points[i]);
    }

    for (int i = 0; i < Points.size(); i++)
    {
        Points[i].x = -Points[i].x;
        Points[i].y = -Points[i].y;
    }
    sort(Points.begin(), Points.end());
    stack<point> stk_low;
    stk_low.push(Points[0]);
    stk_low.push(Points[1]);
    for (int i = 2; i < Points.size(); i++)
    {
        while (stk_low.size() >= 2)
        {
            point p2 = stk_low.top();
            stk_low.pop();
            point p3 = stk_low.top();
            if (ccw(p3, p2, Points[i]))
            {
                stk_low.push(p2);
                break;
            }
        }
        stk_low.push(Points[i]);
    }

    vector<point> CH;
    stk_low.pop();
    while (!stk_low.empty())
    {
        point p = stk_low.top();
        stk_low.pop();
        p.x = -p.x;
        p.y = -p.y;
        CH.push_back(p);
    }
    stk_up.pop();
    while (!stk_up.empty())
    {
        CH.push_back(stk_up.top());
        stk_up.pop();
    }
    reverse(CH.begin(), CH.end());
    return CH;
}

bool isPointInPolygon(const point &q, const vector<point> &poly)
{
    bool c = 0;
    for (int i = 0; i < poly.size(); i++)
    {
        int j = (i + 1) % poly.size();
        if ((poly[i].y <= q.y && q.y < poly[j].y || poly[j].y <= q.y && q.y < poly[i].y) &&
            q.x < poly[i].x + (poly[j].x - poly[i].x) * (q.y - poly[i].y) / (poly[j].y - poly[i].y))
            c = !c;
    }
    return c;
}

bool isPointInConvexHull(const point &q, const vector<point> &ch)
{
    int n = ch.size();
    if (n < 3)
        return false;
    else if (cross(ch[0], q, ch[1]) > EPS)
        return false;
    else if (cross(ch[0], q, ch[n - 1]) < -EPS)
        return false;

    int l = 2, r = n - 1;
    int line = -1;
    while (l <= r)
    {
        int mid = (l + r) >> 1;
        if (cross(ch[0], q, ch[mid]) > -EPS)
        {
            line = mid;
            r = mid - 1;
        }
        else
            l = mid + 1;
    }
    return cross(ch[line - 1], q, ch[line]) < EPS;
}

// Returns the diameter of a convex hull polygon using rotating calipers algorithm
double diameter(vector<point> &ch)
{
    int n = ch.size();
    if (n == 1)
        return 0;
    if (n == 2)
        return dist(ch[0], ch[1]);

    int k = 1;
    double maxDist = 0;
    for (int i = 0; i < n; i++)
    {
        while (true)
        {
            double curDist = dist(ch[i], ch[(k + 1) % n]);
            double nextDist = dist(ch[i], ch[k]);
            if (curDist > nextDist)
            {
                k = (k + 1) % n;
            }
            else
            {
                break;
            }
        }
        maxDist = max(maxDist, dist(ch[i], ch[k]));
    }
    return maxDist;
}

/******************** Strings ********************/

/********* Suffix Array *********/

// Suffix Array
// Given a string, the suffix array is a sorted array of all suffixes of the string.
// The suffix array is a powerful data structure that can be used to solve many string processing problems.
// The suffix array can be constructed in O(n log n) time using the Skew algorithm.
#define MAX_N 200010

class SuffixArray
{
private:
    vi RA; // rank array

    void countingSort(int k)
    {                                       // O(n)
        int maxi = max(300, n);             // up to 255 ASCII chars
        vi c(maxi, 0);                      // clear frequency table
        for (int i = 0; i < n; ++i)         // count the frequency
            ++c[i + k < n ? RA[i + k] : 0]; // of each integer rank
        for (int i = 0, sum = 0; i < maxi; ++i)
        {
            int t = c[i];
            c[i] = sum;
            sum += t;
        }
        vi tempSA(n);
        for (int i = 0; i < n; ++i) // sort SA
            tempSA[c[SA[i] + k < n ? RA[SA[i] + k] : 0]++] = SA[i];
        swap(SA, tempSA); // update SA
    }

    void constructSA()
    { // can go up to 400K chars
        SA.resize(n);
        iota(SA.begin(), SA.end(), 0); // the initial SA
        RA.resize(n);
        for (int i = 0; i < n; ++i)
            RA[i] = T[i]; // initial rankings
        for (int k = 1; k < n; k <<= 1)
        { // repeat log_2 n times
            // this is actually radix sort
            countingSort(k); // sort by 2nd item
            countingSort(0); // stable-sort by 1st item
            vi tempRA(n);
            int r = 0;
            tempRA[SA[0]] = r;          // re-ranking process
            for (int i = 1; i < n; ++i) // compare adj suffixes
                tempRA[SA[i]] =         // same pair => same rank r; otherwise, increase r
                    ((RA[SA[i]] == RA[SA[i - 1]]) && (RA[SA[i] + k] == RA[SA[i - 1] + k])) ? r : ++r;
            swap(RA, tempRA); // update RA
            if (RA[SA[n - 1]] == n - 1)
                break; // nice optimization
        }
    }

    void computeLCP()
    {
        vi Phi(n);
        vi PLCP(n);
        PLCP.resize(n);
        Phi[SA[0]] = -1;            // default value
        for (int i = 1; i < n; ++i) // compute Phi in O(n)
            Phi[SA[i]] = SA[i - 1]; // remember prev suffix
        for (int i = 0, L = 0; i < n; ++i)
        { // compute PLCP in O(n)
            if (Phi[i] == -1)
            {
                PLCP[i] = 0;
                continue;
            } // special case
            while ((i + L < n) && (Phi[i] + L < n) && (T[i + L] == T[Phi[i] + L]))
                L++; // L incr max n times
            PLCP[i] = L;
            L = max(L - 1, 0); // L dec max n times
        }
        LCP.resize(n);
        for (int i = 0; i < n; ++i) // compute LCP in O(n)
            LCP[i] = PLCP[SA[i]];   // restore PLCP
    }

public:
    const char *T; // the input string
    const int n;   // the length of T
    vi SA;         // Suffix Array
    vi LCP;        // of adj sorted suffixes

    SuffixArray(const char *_T, const int _n) : T(_T), n(_n)
    {
        constructSA(); // O(n log n)
        computeLCP();  // O(n)
    }
};

char T[MAX_N];
char P[MAX_N];
char LRS_ans[MAX_N];
char LCS_ans[MAX_N];

// int main()
// {
//     scanf("%s", T);         // read T
//     int n = (int)strlen(T); // count n
//     T[n++] = '$';           // add terminating symbol
//     SuffixArray S(T, n);    // construct SA+LCP

//     printf("T = '%s'\n", T);
//     printf(" i SA[i] LCP[i]   Suffix SA[i]\n");
//     for (int i = 0; i < n; ++i)
//         printf("%2d    %2d    %2d    %s\n", i, S.SA[i], S.LCP[i], T + S.SA[i]);

//     return 0;
// }

/********* KMP algorithm *********/

// KMP algorithm
// Given a string and a pattern, KMP algorithm finds all the occurrences of the pattern in the string
// The algorithm has two main steps:
// 1. Construct the LPS (Longest Prefix Suffix) array
// 2. Search for the pattern in the string
// The time complexity of the algorithm is O(n + m) where n is the length of the string and m is the length of the pattern
string KMP_str; // The string to search in
string KMP_pat; // The pattern to search
vi lps;

// KMP Init
void KMP_init()
{
    int m = KMP_pat.length();
    lps.resize(m + 1, 0);
    lps[0] = -1;
    int i = 0, j = -1;
    while (i < m)
    {
        while (j >= 0 && KMP_pat[i] != KMP_pat[j])
            j = lps[j];
        i++;
        j++;
        cout << i << endl;
        lps[i] = j;
    }
}

// Search a pattern in a string
// Assuming lps is allready initialized with KMP_init
void KMP_search()
{
    int n = KMP_str.length();
    int m = KMP_pat.length();
    int i = 0, j = 0;
    while (i < n)
    {
        while (j >= 0 && KMP_str[i] != KMP_pat[j])
            j = lps[j];
        i++;
        j++;
        if (j == m)
        { // Pattern found
            cout << "The pattern is found at index " << i - j << endl;
            j = lps[j];
        }
    }
}

int main()
{
    KMP_pat = "aba";
    KMP_str = "abababacababac";
    KMP_init();
    KMP_search();
    return 0;
}