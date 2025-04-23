//Name-kaivalya sawant
//PRN-22070521195

#include <stdio.h>
#include <stdlib.h>

#define MAX 100

int graph[MAX][MAX], visited[MAX];

void dfs(int start, int vertices) {
    visited[start] = 1;
    printf("%d ", start);
    for (int i = 0; i < vertices; i++) {
        if (graph[start][i] && !visited[i]) {
            dfs(i, vertices);
        }
    }
}

int main() {
    int vertices, edges, start, u, v;

    printf("Enter the number of vertices: ");
    scanf("%d", &vertices);

    printf("Enter the number of edges: ");
    scanf("%d", &edges);

    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            graph[i][j] = 0;
        }
    }

    printf("Enter the edges (u v):\n");
    for (int i = 0; i < edges; i++) {
        scanf("%d %d", &u, &v);
        graph[u][v] = 1;
        graph[v][u] = 1;
    }

    printf("Enter the starting vertex: ");
    scanf("%d", &start);

    printf("DFS Traversal: ");
    for (int i = 0; i < vertices; i++) visited[i] = 0;
    dfs(start, vertices);
    printf("\n");

    return 0;
}

