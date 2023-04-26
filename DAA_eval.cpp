// C++ code for solving the Longest Common Subsequence (LCS) problem using Dynamic Programming:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

int lcs(string s1, string s2)
{ // function to find LCS using DP
    int m = s1.length();
    int n = s2.length();
    int dp[m + 1][n + 1]; // 2D array to store the DP table

    for (int i = 0; i <= m; i++)
    {
        for (int j = 0; j <= n; j++)
        {
            if (i == 0 || j == 0)
            { // base case
                dp[i][j] = 0;
            }
            else if (s1[i - 1] == s2[j - 1])
            {                                    // when the characters match
                dp[i][j] = dp[i - 1][j - 1] + 1; // increment the LCS length
            }
            else
            {                                               // when the characters don't match
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]); // choose the maximum LCS length so far
            }
        }
    }
    return dp[m][n]; // return the final LCS length
}

int main()
{
    string s1, s2;
    cin >> s1 >> s2;       // input two strings
    int ans = lcs(s1, s2); // function call to find LCS using DP
    cout << ans << endl;   // print the length of LCS
    return 0;
}

// C++ code for solving the Matrix Chain Multiplication problem using Dynamic Programming:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

int matrixChainOrder(int p[], int n)
{                // function to find the minimum number of scalar multiplications using DP
    int m[n][n]; // 2D array to store the DP table

    for (int i = 1; i < n; i++)
    {
        m[i][i] = 0; // setting the diagonal elements to 0
    }

    for (int L = 2; L < n; L++)
    {
        for (int i = 1; i < n - L + 1; i++)
        {
            int j = i + L - 1;
            m[i][j] = INT_MAX;
            for (int k = i; k < j; k++)
            {
                int q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
                if (q < m[i][j])
                {
                    m[i][j] = q;
                }
            }
        }
    }
    return m[1][n - 1]; // return the minimum number of scalar multiplications
}

int main()
{
    int n;
    cin >> n; // input the number of matrices
    int p[n + 1];
    for (int i = 0; i < n + 1; i++)
    {
        cin >> p[i]; // input the dimensions of matrices
    }
    int ans = matrixChainOrder(p, n + 1); // function call to find the minimum number of scalar multiplications using DP
    cout << ans << endl;                  // print the minimum number of scalar multiplications
    return 0;
}

// C++ code for solving the 0/1 Knapsack problem using Dynamic Programming:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

int knapSack(int W, int wt[], int val[], int n)
{                        // function to find the maximum value that can be obtained using DP
    int K[n + 1][W + 1]; // 2D array to store the DP table

    for (int i = 0; i <= n; i++)
    {
        for (int w = 0; w <= W; w++)
        {
            if (i == 0 || w == 0)
            {
                K[i][w] = 0; // setting the base case values to 0
            }
            else if (wt[i - 1] <= w)
            {
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w]); // calculating the maximum value
            }
            else
            {
                K[i][w] = K[i - 1][w]; // copying the value from the previous row
            }
        }
    }
    return K[n][W]; // return the maximum value that can be obtained
}

int main()
{
    int n, W;
    cin >> n >> W; // input the number of items and the maximum weight of knapsack
    int val[n], wt[n];
    for (int i = 0; i < n; i++)
    {
        cin >> val[i] >> wt[i]; // input the values and weights of items
    }
    int ans = knapSack(W, wt, val, n); // function call to find the maximum value that can be obtained using DP
    cout << ans << endl;               // print the maximum value that can be obtained
    return 0;
}

// C++ code for solving the Optimal Binary Search Tree problem using Dynamic Programming:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

float optimalBST(float keys[], float freq[], int n)
{                             // function to find the cost of optimal BST using DP
    float cost[n + 1][n + 1]; // 2D array to store the DP table
    for (int i = 0; i <= n; i++)
    {
        for (int j = 0; j <= n; j++)
        {
            cost[i][j] = 0; // initializing all the elements of the DP table to 0
        }
    }
    for (int i = 0; i < n; i++)
    {
        cost[i][i] = freq[i]; // setting the diagonal elements to frequency of ith key
    }
    for (int L = 2; L <= n; L++)
    {
        for (int i = 0; i <= n - L + 1; i++)
        {
            int j = i + L - 1;
            cost[i][j] = FLT_MAX; // setting the initial value to infinity
            for (int r = i; r <= j; r++)
            {
                float c = ((r > i) ? cost[i][r - 1] : 0) + ((r < j) ? cost[r + 1][j] : 0) + accumulate(freq + i, freq + j + 1, 0);
                if (c < cost[i][j])
                {
                    cost[i][j] = c;
                }
            }
        }
    }
    return cost[0][n - 1]; // return the cost of optimal BST
}

int main()
{
    int n;
    cin >> n; // input the number of keys
    float keys[n], freq[n];
    for (int i = 0; i < n; i++)
    {
        cin >> keys[i] >> freq[i]; // input the keys and their frequencies
    }
    float ans = optimalBST(keys, freq, n); // function call to find the cost of optimal BST using DP
    cout << ans << endl;                   // print the cost of optimal BST
    return 0;
}

// C++ code for solving the Coin Exchange problem using Dynamic Programming:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

int coinExchange(int coins[], int n, int sum)
{                           // function to find the minimum number of coins required to make sum using DP
    int dp[n + 1][sum + 1]; // 2D array to store the DP table
    for (int i = 0; i <= n; i++)
    {
        for (int j = 0; j <= sum; j++)
        {
            if (j == 0)
            {
                dp[i][j] = 0; // initializing the first column to 0
            }
            else if (i == 0)
            {
                dp[i][j] = INT_MAX; // initializing the first row to infinity
            }
            else if (coins[i - 1] <= j)
            {
                dp[i][j] = min(dp[i][j - coins[i - 1]] + 1, dp[i - 1][j]); // filling the DP table using the recurrence relation
            }
            else
            {
                dp[i][j] = dp[i - 1][j]; // copying the value from the previous row if the coin value is greater than the sum
            }
        }
    }
    if (dp[n][sum] == INT_MAX)
    {
        return -1; // if it is not possible to make sum using the given coins, return -1
    }
    return dp[n][sum]; // return the minimum number of coins required to make sum
}

int main()
{
    int n, sum;
    cin >> n >> sum; // input the number of coins and the sum to be made
    int coins[n];
    for (int i = 0; i < n; i++)
    {
        cin >> coins[i]; // input the values of the coins
    }
    int ans = coinExchange(coins, n, sum); // function call to find the minimum number of coins required to make sum using DP
    cout << ans << endl;                   // print the minimum number of coins required to make sum
    return 0;
}

// C++ code for solving the Rod Cutting problem using Dynamic Programming:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

int rodCutting(int price[], int n)
{                  // function to find the maximum profit using DP
    int dp[n + 1]; // 1D array to store the DP table
    dp[0] = 0;     // initializing the first element to 0
    for (int i = 1; i <= n; i++)
    {
        int max_val = INT_MIN;
        for (int j = 0; j < i; j++)
        {
            max_val = max(max_val, price[j] + dp[i - j - 1]); // filling the DP table using the recurrence relation
        }
        dp[i] = max_val; // storing the maximum profit in the DP table
    }
    return dp[n]; // return the maximum profit
}

int main()
{
    int n;
    cin >> n; // input the length of the rod
    int price[n];
    for (int i = 0; i < n; i++)
    {
        cin >> price[i]; // input the price of each piece of the rod of length i+1
    }
    int ans = rodCutting(price, n); // function call to find the maximum profit using DP
    cout << ans << endl;            // print the maximum profit
    return 0;
}

// C++ code for solving the Weighted Job Sequencing problem using Dynamic Programming:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

struct Job
{ // struct to store the job details
    int start, finish, profit;
};

bool compare(Job a, Job b)
{ // function to compare jobs based on finish time
    return a.finish < b.finish;
}

int findLastNonConflictingJob(Job arr[], int n, int index)
{ // function to find the last non-conflicting job
    for (int i = index - 1; i >= 0; i--)
    {
        if (arr[i].finish <= arr[index].start)
        {
            return i;
        }
    }
    return -1;
}

int weightedJobSequencing(Job arr[], int n)
{                                // function to find the maximum profit using DP
    sort(arr, arr + n, compare); // sort the jobs based on finish time
    int dp[n];                   // 1D array to store the DP table
    dp[0] = arr[0].profit;       // initialize the first element of the array
    for (int i = 1; i < n; i++)
    {
        int inclProfit = arr[i].profit;               // profit if the current job is included
        int l = findLastNonConflictingJob(arr, n, i); // find the last non-conflicting job
        if (l != -1)
        {
            inclProfit += dp[l]; // add the profit of the last non-conflicting job
        }
        dp[i] = max(inclProfit, dp[i - 1]); // store the maximum profit in the DP table
    }
    return dp[n - 1]; // return the maximum profit
}

int main()
{
    int n;
    cin >> n; // input the number of jobs
    Job arr[n];
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i].start >> arr[i].finish >> arr[i].profit; // input the start time, finish time and profit of each job
    }
    int ans = weightedJobSequencing(arr, n); // function call to find the maximum profit using DP
    cout << ans << endl;                     // print the maximum profit
    return 0;
}

// C++ code for solving the N-Queens problem using backtracking:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

int board[10][10]; // 2D array to store the state of the chess board
int n;             // size of the chess board

void printSolution()
{ // function to print the solution
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << board[i][j] << " "; // print the state of each cell of the board
        }
        cout << endl;
    }
}

bool isSafe(int row, int col)
{ // function to check if it is safe to place a queen at the given position
    for (int i = 0; i < row; i++)
    { // check if there is any queen in the same column
        if (board[i][col])
        {
            return false;
        }
    }
    for (int i = row, j = col; i >= 0 && j >= 0; i--, j--)
    { // check if there is any queen in the diagonal to the left
        if (board[i][j])
        {
            return false;
        }
    }
    for (int i = row, j = col; i >= 0 && j < n; i--, j++)
    { // check if there is any queen in the diagonal to the right
        if (board[i][j])
        {
            return false;
        }
    }
    return true; // return true if it is safe to place a queen
}

bool solveNQueen(int row)
{ // function to solve the N-Queens problem using backtracking
    if (row == n)
    { // if all the queens are placed successfully, return true
        return true;
    }
    for (int col = 0; col < n; col++)
    { // try placing a queen in each column of the current row
        if (isSafe(row, col))
        {                        // check if it is safe to place a queen at the current position
            board[row][col] = 1; // place the queen at the current position
            if (solveNQueen(row + 1))
            { // recursively solve the N-Queens problem for the next row
                return true;
            }
            board[row][col] = 0; // backtrack and remove the queen from the current position
        }
    }
    return false; // if a queen cannot be placed in any column of the current row, return false
}

int main()
{
    cin >> n; // input the size of the chess board
    if (solveNQueen(0))
    {                    // function call to solve the N-Queens problem using backtracking
        printSolution(); // print the solution if it exists
    }
    else
    {
        cout << "Solution does not exist." << endl; // print message if the solution does not exist
    }
    return 0;
}

// C++ code for solving the Sum of Subsets problem using backtracking:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

int arr[10];                 // array to store the input elements
int n;                       // size of the input array
int sum;                     // target sum
bool solutionExists = false; // variable to check if a solution exists

void printSolution(vector<int> &v)
{ // function to print the solution
    cout << "{ ";
    for (int i = 0; i < v.size(); i++)
    {
        cout << v[i] << " "; // print the elements that form the subset
    }
    cout << "}" << endl;
}

void solveSubsetSum(int index, int currentSum, vector<int> &subset)
{ // function to solve the Sum of Subsets problem using backtracking
    if (currentSum == sum)
    { // if the target sum is reached, print the subset and return
        printSolution(subset);
        solutionExists = true;
        return;
    }
    for (int i = index; i < n; i++)
    { // try adding each element of the input array one by one
        if (currentSum + arr[i] <= sum)
        {                                                       // check if adding the current element to the sum does not exceed the target sum
            subset.push_back(arr[i]);                           // add the current element to the subset
            solveSubsetSum(i + 1, currentSum + arr[i], subset); // recursively solve the problem for the next element
            subset.pop_back();                                  // backtrack and remove the current element from the subset
        }
    }
}

int main()
{
    cin >> n; // input the size of the input array
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i]; // input the elements of the input array
    }
    cin >> sum;                   // input the target sum
    vector<int> subset;           // vector to store the elements that form the subset
    solveSubsetSum(0, 0, subset); // function call to solve the Sum of Subsets problem using backtracking
    if (!solutionExists)
    {
        cout << "Solution does not exist." << endl; // print message if the solution does not exist
    }
    return 0;
}

// C++ code for solving the Graph Coloring problem using backtracking:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

int graph[10][10]; // adjacency matrix to represent the graph
int n;             // number of vertices in the graph
int m;             // number of colors available for coloring the vertices
int color[10];     // array to store the color assigned to each vertex

bool isSafe(int v, int c)
{ // function to check if it is safe to assign color c to vertex v
    for (int i = 0; i < n; i++)
    {
        if (graph[v][i] && c == color[i])
        { // if there exists an edge between vertex v and i and the color assigned to vertex i is c, return false
            return false;
        }
    }
    return true; // otherwise, return true
}

void printColors()
{ // function to print the colors assigned to each vertex
    cout << "Colors assigned to vertices:" << endl;
    for (int i = 0; i < n; i++)
    {
        cout << "Vertex " << i << ": Color " << color[i] << endl;
    }
}

bool graphColoring(int v)
{ // function to solve the Graph Coloring problem using backtracking
    if (v == n)
    { // if all the vertices have been colored, print the colors and return true
        printColors();
        return true;
    }
    for (int c = 1; c <= m; c++)
    { // try assigning each color one by one to the current vertex
        if (isSafe(v, c))
        {                 // check if it is safe to assign color c to vertex v
            color[v] = c; // assign color c to vertex v
            if (graphColoring(v + 1))
            { // recursively solve the problem for the next vertex
                return true;
            }
            color[v] = 0; // backtrack and remove the color assigned to vertex v
        }
    }
    return false; // if no color can be assigned to the current vertex, return false
}

int main()
{
    cin >> n; // input the number of vertices in the graph
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cin >> graph[i][j]; // input the adjacency matrix
        }
    }
    cin >> m; // input the number of colors available for coloring the vertices
    if (!graphColoring(0))
    {
        cout << "Solution does not exist." << endl; // print message if a solution does not exist
    }
    return 0;
}

// C++ code for solving the 0/1 Knapsack problem using backtracking:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

int n;            // number of items
int w[10];        // weight of each item
int p[10];        // profit of each item
int W;            // maximum weight that the knapsack can hold
bool include[10]; // array to store whether an item is included in the knapsack or not
int maxProfit;    // variable to store the maximum profit obtained

void knapsack(int i, int profit, int weight)
{ // function to solve the 0/1 Knapsack problem using backtracking
    if (i == n)
    { // if all items have been considered
        if (profit > maxProfit)
        {                       // if the current profit is greater than the maximum profit obtained so far
            maxProfit = profit; // update the maximum profit
            for (int j = 0; j < n; j++)
            {
                cout << include[j] << " "; // print the items included in the knapsack
            }
            cout << endl;
        }
        return;
    }
    if (weight + w[i] <= W)
    {                                                  // if the current item can be included in the knapsack
        include[i] = true;                             // include the item
        knapsack(i + 1, profit + p[i], weight + w[i]); // recursively solve the problem for the next item
        include[i] = false;                            // backtrack and remove the item
    }
    knapsack(i + 1, profit, weight); // solve the problem by excluding the current item
}

int main()
{
    cin >> n; // input the number of items
    for (int i = 0; i < n; i++)
    {
        cin >> w[i] >> p[i]; // input the weight and profit of each item
    }
    cin >> W;                                        // input the maximum weight that the knapsack can hold
    maxProfit = 0;                                   // initialize the maximum profit to 0
    knapsack(0, 0, 0);                               // solve the 0/1 Knapsack problem using backtracking
    cout << "Maximum profit: " << maxProfit << endl; // print the maximum profit obtained
    return 0;
}

// C++ code for solving the Hamiltonian Cycle problem using backtracking:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

int n;            // number of vertices in the graph
int g[10][10];    // adjacency matrix to represent the graph
int path[10];     // array to store the current path
bool visited[10]; // array to store whether a vertex has been visited or not
bool hamiltonian; // variable to store whether a Hamiltonian Cycle exists or not

void hamiltonianCycle(int k)
{ // function to solve the Hamiltonian Cycle problem using backtracking
    if (k == n)
    { // if all vertices have been included in the path
        if (g[path[n - 1]][path[0]] == 1)
        {                       // if there is an edge from the last vertex to the first vertex
            hamiltonian = true; // a Hamiltonian Cycle exists
            for (int i = 0; i < n; i++)
            {
                cout << path[i] << " "; // print the Hamiltonian Cycle
            }
            cout << endl;
        }
        return;
    }
    for (int i = 0; i < n; i++)
    {
        if (g[path[k - 1]][i] == 1 && !visited[i])
        {                            // if there is an edge between the current vertex and the next vertex, and the next vertex has not been visited
            path[k] = i;             // add the next vertex to the path
            visited[i] = true;       // mark the next vertex as visited
            hamiltonianCycle(k + 1); // recursively solve the problem for the next vertex
            visited[i] = false;      // backtrack and remove the next vertex from the path
        }
    }
}

int main()
{
    cin >> n; // input the number of vertices in the graph
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cin >> g[i][j]; // input the adjacency matrix of the graph
        }
    }
    hamiltonian = false; // initialize the Hamiltonian Cycle to false
    path[0] = 0;         // add the first vertex to the path
    visited[0] = true;   // mark the first vertex as visited
    hamiltonianCycle(1); // solve the Hamiltonian Cycle problem using backtracking
    if (!hamiltonian)
    { // if a Hamiltonian Cycle does not exist
        cout << "No Hamiltonian Cycle exists" << endl;
    }
    return 0;
}

// C++ code for solving the Printing N-Queen Solutions problem using backtracking:

#include <bits/stdc++.h> // header file to include all standard libraries
using namespace std;

int n;       // number of queens
int col[10]; // array to store the column number of the queens in each row
int cnt;     // variable to store the number of solutions found

bool isSafe(int row, int col)
{ // function to check whether it is safe to place a queen at a particular position
    for (int i = 0; i < row; i++)
    {
        if (col == col[i] || abs(col - col[i]) == abs(row - i))
        {                 // if another queen is present in the same column or diagonal
            return false; // it is not safe to place a queen at this position
        }
    }
    return true; // it is safe to place a queen at this position
}

void nQueen(int row)
{ // function to solve the Printing N-Queen Solutions problem using backtracking
    if (row == n)
    {          // if all queens have been placed on the board
        cnt++; // increment the number of solutions found
        cout << "Solution " << cnt << ":" << endl;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (col[i] == j)
                {
                    cout << "Q "; // print Q for queen
                }
                else
                {
                    cout << ". "; // print . for empty cell
                }
            }
            cout << endl;
        }
        cout << endl;
        return;
    }
    for (int i = 0; i < n; i++)
    {
        if (isSafe(row, i))
        {                    // if it is safe to place a queen at this position
            col[row] = i;    // place the queen at this position
            nQueen(row + 1); // recursively solve the problem for the next row
        }
    }
}

int main()
{
    cin >> n;  // input the number of queens
    cnt = 0;   // initialize the number of solutions found to zero
    nQueen(0); // solve the Printing N-Queen Solutions problem using backtracking
    if (cnt == 0)
    { // if no solutions were found
        cout << "No solutions exist" << endl;
    }
    return 0;
}

// C++ code without vectors or any additional information to solve the Euler graph problem using the Hierholzer algorithm:

#include <iostream>
#include <stack>
#include <cstring>
#define MAXN 1001
#define MAXM 100001

using namespace std;

int head[MAXN], ver[MAXM], Next[MAXM];
bool vis[MAXM];
int cnt, m, n;

void add(int a, int b)
{
    ver[++cnt] = b;
    Next[cnt] = head[a];
    head[a] = cnt;
}

void euler(int x)
{
    for (int i = head[x]; i; i = Next[i])
    {
        if (!vis[i])
        {
            vis[i] = 1;
            euler(ver[i]);
            cout << x << "->" << ver[i] << " ";
        }
    }
}

int main()
{
    memset(head, 0, sizeof(head));
    memset(vis, 0, sizeof(vis));
    cnt = 0;

    cin >> n >> m;
    for (int i = 1; i <= m; i++)
    {
        int a, b;
        cin >> a >> b;
        add(a, b);
        add(b, a);
    }

    euler(1);
    return 0;
}

// C++ code without vectors or any additional information to solve the Hamiltonian graph problem using backtracking:

#include <iostream>
#include <cstring>
#define MAXN 1001

using namespace std;

int n, m, ans = 0;
int graph[MAXN][MAXN];
int visited[MAXN];

void hamiltonian(int cur, int depth)
{
    visited[cur] = depth;
    if (depth == n)
    {
        ans = 1;
        return;
    }
    for (int i = 1; i <= n; i++)
    {
        if (graph[cur][i] && !visited[i])
        {
            hamiltonian(i, depth + 1);
            if (ans)
            {
                return;
            }
        }
    }
    visited[cur] = 0;
}

int main()
{
    memset(graph, 0, sizeof(graph));
    memset(visited, 0, sizeof(visited));

    cin >> n >> m;
    for (int i = 1; i <= m; i++)
    {
        int u, v;
        cin >> u >> v;
        graph[u][v] = 1;
        graph[v][u] = 1;
    }

    for (int i = 1; i <= n; i++)
    {
        hamiltonian(i, 1);
        if (ans)
        {
            cout << "Hamiltonian path exists." << endl;
            return 0;
        }
    }
    cout << "Hamiltonian path does not exist." << endl;
    return 0;
}

// C++ code for implementing topological sort using Kahn's algorithm:

#include <iostream>
#include <vector>
#include <queue>

using namespace std;

// Function to perform topological sorting
void topologicalSort(vector<int> adj[], int V)
{
    // Create an array to store the in-degree of each vertex
    vector<int> in_degree(V, 0);

    // Traverse the graph and update the in-degree of each vertex
    for (int u = 0; u < V; u++)
    {
        for (int v : adj[u])
        {
            in_degree[v]++;
        }
    }

    // Create a queue and enqueue all vertices with in-degree 0
    queue<int> q;
    for (int u = 0; u < V; u++)
    {
        if (in_degree[u] == 0)
        {
            q.push(u);
        }
    }

    // Initialize a counter to keep track of visited vertices
    int count = 0;

    // Create a vector to store the topological ordering of vertices
    vector<int> top_order;

    // Process vertices until the queue becomes empty
    while (!q.empty())
    {
        // Dequeue a vertex from the queue
        int u = q.front();
        q.pop();

        // Add the vertex to the topological ordering vector
        top_order.push_back(u);

        // Traverse all the adjacent vertices of u
        for (int v : adj[u])
        {
            // Decrement the in-degree of each adjacent vertex
            in_degree[v]--;

            // If the in-degree of a vertex becomes 0, enqueue it
            if (in_degree[v] == 0)
            {
                q.push(v);
            }
        }

        // Increment the counter
        count++;
    }

    // Check if there was a cycle in the graph
    if (count != V)
    {
        cout << "There exists a cycle in the graph" << endl;
        return;
    }

    // Print the topological ordering of vertices
    for (int u : top_order)
    {
        cout << u << " ";
    }
}

int main()
{
    int V, E;
    cout << "Enter the number of vertices in the graph: ";
    cin >> V;
    cout << "Enter the number of edges in the graph: ";
    cin >> E;

    // Create an adjacency list to represent the graph
    vector<int> adj[V];
    for (int i = 0; i < E; i++)
    {
        int u, v;
        cout << "Enter the endpoints of edge " << i + 1 << ": ";
        cin >> u >> v;
        adj[u].push_back(v);
    }

    // Perform topological sorting
    topologicalSort(adj, V);

    return 0;
}

// C++ code for topological sort using Kahn's algorithm, without using vectors:

#include <iostream>
#include <queue>
#include <list>
#include <cstring>

#define MAX_NODES 10001 // maximum number of nodes

using namespace std;

int indegree[MAX_NODES];  // array to store indegrees of all nodes
list<int> adj[MAX_NODES]; // adjacency list to store graph

void topologicalSort(int n)
{
    queue<int> q;           // create a queue to store nodes with indegree 0
    int visited[MAX_NODES]; // array to keep track of visited nodes
    memset(visited, 0, sizeof visited);

    for (int i = 1; i <= n; i++)
    { // loop through all nodes
        if (indegree[i] == 0)
        {                   // if node has indegree 0
            q.push(i);      // add it to the queue
            visited[i] = 1; // mark it as visited
        }
    }

    while (!q.empty())
    {                      // while queue is not empty
        int u = q.front(); // get the front node from the queue
        q.pop();           // remove the front node from the queue
        cout << u << " ";  // print the node

        for (auto v : adj[u])
        {                  // loop through all the neighbors of node u
            indegree[v]--; // reduce the indegree of the neighbor
            if (indegree[v] == 0 && !visited[v])
            {                   // if neighbor's indegree becomes 0 and it has not been visited before
                q.push(v);      // add it to the queue
                visited[v] = 1; // mark it as visited
            }
        }
    }
}

int main()
{
    int n, m;
    cin >> n >> m; // read number of nodes and edges

    for (int i = 0; i < m; i++)
    {
        int u, v;
        cin >> u >> v;       // read edge (u, v)
        adj[u].push_back(v); // add edge (u, v) to the adjacency list
        indegree[v]++;       // increment indegree of node v
    }

    topologicalSort(n); // call topological sort function

    return 0;
}

// C++ code to perform topological sort on a given directed acyclic graph using Depth First Search (DFS) algorithm without using vectors:

#include <bits/stdc++.h>
using namespace std;

// Define the maximum number of vertices in the graph
#define MAXN 100001

// Adjacency list to store graph
list<int> adj[MAXN];

// Function to add edge to the graph
void addEdge(int u, int v)
{
    adj[u].push_back(v);
}

// Recursive function to perform DFS
void DFS(int u, stack<int> &S, bool visited[])
{
    visited[u] = true;

    // Traverse all the adjacent vertices of vertex u
    for (auto i = adj[u].begin(); i != adj[u].end(); ++i)
    {
        int v = *i;

        // If v is not visited yet, call DFS recursively
        if (!visited[v])
            DFS(v, S, visited);
    }

    // Push vertex u onto the stack
    S.push(u);
}

// Function to perform topological sort using DFS
void topologicalSortDFS(int N)
{
    // Create a stack to store the sorted order
    stack<int> S;

    // Initialize all vertices as not visited
    bool visited[MAXN] = {false};

    // Call the recursive DFS function for each unvisited vertex
    for (int i = 1; i <= N; i++)
    {
        if (!visited[i])
            DFS(i, S, visited);
    }

    // Print the contents of the stack in LIFO order
    while (!S.empty())
    {
        cout << S.top() << " ";
        S.pop();
    }
}

int main()
{
    // Define the number of vertices and edges in the graph
    int N = 6, M = 6;

    // Add edges to the graph
    addEdge(1, 2);
    addEdge(1, 4);
    addEdge(2, 3);
    addEdge(2, 4);
    addEdge(3, 5);
    addEdge(6, 4);

    // Perform topological sort
    topologicalSortDFS(N);

    return 0;
}

// C++ code for solving the Maximum Flow problem using the Ford-Fulkerson algorithm:

#include <bits/stdc++.h>
using namespace std;

const int N = 1010;  // maximum number of vertices
const int INF = 1e9; // infinity

int n, s, t; // number of vertices, source and sink
int c[N][N]; // capacity matrix
int f[N][N]; // flow matrix
int p[N];    // parent array for path finding

// Function to find an augmented path using BFS
bool bfs()
{
    bool visited[N] = {0}; // visited array
    queue<int> q;          // queue for BFS
    q.push(s);             // start from the source
    visited[s] = true;
    p[s] = -1;
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        for (int v = 0; v < n; v++)
        {
            if (!visited[v] && c[u][v] > f[u][v])
            {
                q.push(v);
                visited[v] = true;
                p[v] = u;
            }
        }
    }
    return visited[t];
}

// Function to find the maximum flow using Ford-Fulkerson algorithm
int fordFulkerson()
{
    int maxFlow = 0;
    while (bfs())
    {
        int pathFlow = INF;
        for (int v = t; v != s; v = p[v])
        {
            int u = p[v];
            pathFlow = min(pathFlow, c[u][v] - f[u][v]);
        }
        for (int v = t; v != s; v = p[v])
        {
            int u = p[v];
            f[u][v] += pathFlow;
            f[v][u] -= pathFlow;
        }
        maxFlow += pathFlow;
    }
    return maxFlow;
}

int main()
{
    // Input number of vertices, source and sink
    cin >> n >> s >> t;
    // Input capacity matrix
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cin >> c[i][j];
        }
    }
    int maxFlow = fordFulkerson();
    cout << "Maximum flow: " << maxFlow << endl;
    return 0;
}

// C++ code for the weighted min-cut problem using graphs:

#include <iostream>
#include <cstring>
#include <climits>
#define V 4 // number of vertices

// function to find the vertex with maximum weight in the given cut
int findMaxWeightVertex(bool visited[], int weight[])
{
    int maxWeight = INT_MIN, maxWeightVertex;
    for (int v = 0; v < V; v++)
    {
        if (!visited[v] && weight[v] > maxWeight)
        {
            maxWeight = weight[v];
            maxWeightVertex = v;
        }
    }
    return maxWeightVertex;
}

// function to print the minimum cut
void printMinCut(int graph[V][V], int cut[])
{
    std::cout << "Minimum cut:\n";
    for (int u = 0; u < V; u++)
    {
        for (int v = u + 1; v < V; v++)
        {
            if (cut[u] != cut[v])
            {
                std::cout << u << " - " << v << std::endl;
            }
        }
    }
}

// function to find the minimum cut in the given graph
void findMinCut(int graph[V][V])
{
    int weight[V];                           // weight of each vertex
    bool visited[V];                         // track visited vertices
    memset(visited, false, sizeof(visited)); // initialize all vertices as unvisited

    // initialize all weights as the weight of their first edge
    for (int v = 0; v < V; v++)
    {
        weight[v] = graph[0][v];
    }

    int cut[V];  // track the cut
    cut[0] = -1; // vertex 0 is in the first cut

    // iterate V-1 times to find all vertices in the cut
    for (int i = 0; i < V - 1; i++)
    {
        int u = findMaxWeightVertex(visited, weight); // find the vertex with maximum weight
        visited[u] = true;                            // mark the vertex as visited

        // update the weight of the adjacent vertices
        for (int v = 0; v < V; v++)
        {
            if (graph[u][v] && !visited[v] && graph[u][v] > weight[v])
            {
                weight[v] = graph[u][v];
                cut[v] = u; // update the cut
            }
        }
    }

    printMinCut(graph, cut);
}

// driver function
int main()
{
    int graph[V][V] = {{0, 3, 1, 6},
                       {3, 0, 5, 1},
                       {1, 5, 0, 2},
                       {6, 1, 2, 0}};
    findMinCut(graph);
    return 0;
}

// C++ code for solving the weighted max-cut problem using graphs without vectors:

#include <iostream>
#include <cstring>
#include <queue>

using namespace std;

const int MAX_N = 100;
const int MAX_M = 10000;
const int INF = 1e9;

int head[MAX_N], nxt[MAX_M], ver[MAX_M], w[MAX_M], tot = 1;
int dis[MAX_N], cur[MAX_N], S, T, n, m, cnt = 1;
bool vis[MAX_N];

void add(int x, int y, int z)
{
    ver[++tot] = y;
    w[tot] = z;
    nxt[tot] = head[x];
    head[x] = tot;
}

bool bfs()
{
    memset(dis, -1, sizeof(dis));
    queue<int> q;
    q.push(S);
    dis[S] = 0;
    vis[S] = true;
    while (!q.empty())
    {
        int x = q.front();
        q.pop();
        for (int i = head[x]; i; i = nxt[i])
        {
            int y = ver[i], z = w[i];
            if (dis[y] == -1 && z)
            {
                dis[y] = dis[x] + 1;
                vis[y] = true;
                q.push(y);
            }
        }
    }
    return dis[T] != -1;
}

int dfs(int x, int limit)
{
    if (x == T || !limit)
        return limit;
    int flow = 0, f;
    for (int &i = cur[x]; i; i = nxt[i])
    {
        int y = ver[i], z = w[i];
        if (dis[y] == dis[x] + 1 && (f = dfs(y, min(z, limit - flow))))
        {
            flow += f;
            w[i] -= f;
            w[i ^ 1] += f;
            if (flow == limit)
                break;
        }
    }
    if (!flow)
        dis[x] = -1;
    return flow;
}

int dinic()
{
    int maxflow = 0, flow;
    while (bfs())
    {
        memset(cur, 0, sizeof(cur));
        while (flow = dfs(S, INF))
            maxflow += flow;
        memset(vis, false, sizeof(vis));
    }
    return maxflow;
}

int main()
{
    cin >> n >> m;
    S = 1;
    T = n;
    for (int i = 1; i <= m; i++)
    {
        int x, y, z;
        cin >> x >> y >> z;
        add(x, y, z);
        add(y, x, 0); // Add reverse edge with 0 weight
        add(y, x, z);
        add(x, y, 0); // Add reverse edge with 0 weight
    }
    cout << dinic() << endl;
    return 0;
}