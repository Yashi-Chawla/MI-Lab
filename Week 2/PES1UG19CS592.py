"""
You can create any other helper funtions.
Do not modify the given functions
"""
import queue

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path = []
    # TODO
    n = len(cost)                                               
    visited = [0 for i in range(n)]                             # visited (0 - not visited, 1 - visited)
    pq = queue.PriorityQueue()            

    # Queue structure (estimated_cost, start_point, path_till_start_node, 0)
    pq.put((heuristic[start_point], [start_point], start_point, 0))

    while(pq.qsize() != 0):
    # if empty return path
    # no? then check for visited and goal
        # Pop 
        estimated_total_cost,path_till_node,node,node_cost=pq.get()

        if visited[node] == 0:
            visited[node] = 1

            if node in goals: #n is a goal, stop and report success
                path= path_till_node
                return path

            for next_node in range(1, n):
                temp=cost[node]
                if temp[next_node] > 0 and visited[next_node] == 0:
                    #total cost= g(n)+cost(n,n')
                    total_cost_till_node = node_cost + temp[next_node]
                    #estimated cost =total cost+h(n')
                    estimated_total_cost = total_cost_till_node + heuristic[next_node]
                    #add next
                    path_till_next_node=[*path_till_node,next_node]
                    # Insert (estimated total, (next_node, path_till_next_node, total cost till neighbour nodes)) into priority queue
                    pq.put((estimated_total_cost, path_till_next_node, next_node, total_cost_till_node))

    # return empty list if path to goal nodes is not found
    path=path_till_node
    return path


def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path = []
    # TODO
    n = len(cost)                               
    visited = [0 for i in range(n)]                #visited (0 - not visited, 1 - visited)
    tr_stack=queue.LifoQueue()                     #traversal stack
    tr_stack.put((start_point, [start_point]))
    while(tr_stack.qsize()!=0):
        node, path_till_node=tr_stack.get()         #pop
        if visited[node]==0:
            visited[node]=1
            if node in goals:                       #traversal completed, return final path
                path=path_till_node
                return path
            temp=cost[node]
            for next_node in range(n-1,0,-1):       #finding neighbour nodes to be traversed
                if temp[next_node]>0:                   
                    if visited[next_node]==0:
                        path_till_next_node=[*path_till_node,next_node]
                        tr_stack.put((next_node,path_till_next_node))
    path=path_till_node
    return path
