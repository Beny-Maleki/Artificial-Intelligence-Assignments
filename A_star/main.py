import time
from heapq import heappush, heappop
from timeit import Timer

class Node:
    visited_states = set()  # set of visited states
    id_to_node_map = {}
    ID = 0

    # a function to delete all nodes from memory
    @staticmethod
    def clear():
        Node.visited_states = set()
        Node.id_to_node_map = {}
        Node.ID = 0

    ## dest and n will be fixed in main function
    dest = None
    n = -1

    def __init__(self, state, g_val, action_dir, action_index, parent):
        Node.ID += 1
        self.id = Node.ID
        self.parent = parent
        self.state = state
        self.g_val = g_val
        self.h_val = self.h(Node.dest, Node.n)
        self.f_val = g_val + self.h_val
        self.action_dir = action_dir
        self.action_index = action_index
        Node.id_to_node_map[self.id] = self


    def h(self, dest, n):
        sum_min_manhattan = 0
        for i in range(n):
            for j in range(n):
                # dest[self.state[i][j]][0] is the x coordinate of the number self.state[i][j] in the destination state
                h_dist = abs(i - dest[self.state[i][j]][0])
                # dest[self.state[i][j]][1] is the y coordinate of the number self.state[i][j] in the destination state
                v_dist = abs(j - dest[self.state[i][j]][1])
                complement_h_dist = self.n - h_dist
                complement_v_dist = self.n - v_dist

                if complement_h_dist < abs(h_dist):
                    h_dist = complement_h_dist
                if complement_v_dist < abs(v_dist):
                    v_dist = complement_v_dist

                sum_min_manhattan += v_dist + h_dist
        return sum_min_manhattan * 2


    def check_all_next_states(self, n, min_heap):
        # Checks all 4n next possible states (some of them may be repetitious)
        for i in range(n):
            self.check_next_state(min_heap, i, 'R', True)
            self.check_next_state(min_heap, i, 'L', True)
            self.check_next_state(min_heap, i, 'U', False)
            self.check_next_state(min_heap, i, 'D', False)

    def check_next_state(self, min_heap, sel_index, direction, is_row):
        next_state = self.get_next_state(sel_index, is_row, direction)
        hashable_next_state = hashable_state(next_state)
        ## As h(n) is fixed for each state, no need for correction of in-queue states!
        # If the next state is already visited or in-queue, ignore it
        # Otherwise, add it to the queueُ
        if not (hashable_next_state in Node.visited_states):
            # Creating node of new state for first&last time
            child = Node(next_state, self.g_val + 1, direction, sel_index, self)
            # Adding node to queue and marking it as in-queue
            min_heap.insert_node(child)


    def get_next_state(self, sel_index, is_row, direction):
        state = self.state.copy()  # changing reference of state
        if is_row:
            if direction == 'L':
                shift_negative(sel_index, state)
            if direction == 'R':
                shift_positive(sel_index, state)
        else:
            transposed_state = [list(x) for x in zip(*state)]
            if direction == 'U':
                shift_negative(sel_index, transposed_state)
            elif direction == 'D':
                shift_positive(sel_index, transposed_state)

            state = [list(x) for x in zip(*transposed_state)]
        return state


def shift_positive(sel_index, state):
    state[sel_index] = [state[sel_index][-1]] + state[sel_index][:-1]


def shift_negative(sel_index, state):
    state[sel_index] = state[sel_index][1:] + [state[sel_index][0]]


def hashable_state(state):
    return str(state)


class MinHeap:
    def __init__(self):
        self.heap = []

    def insert_node(self, node):
        # we push tuple of (objective function, str of state) into heap
        heappush(self.heap, (node.f_val, hashable_state(node.state), node.id))

    def pop_min(self):
        return heappop(self.heap)


def a_star(source):
    # A* search
    # 1. Create a priority queue
    min_heap = MinHeap()
    # 2. Create source node and add it to the queue
    source_node = Node(source, 0, None, 0, None)
    min_heap.insert_node(source_node)
    # 3. While queue is not empty
    num_expanded_nodes = 0
    while min_heap.heap:
        # 4. Pop the node with the lowest cost (expanding queue head)
        # min_heap.pop_min()[1] is the string format of state
        popped_tuple = min_heap.pop_min()
        node_state_string = popped_tuple[1]
        node_id = popped_tuple[2]
        node = Node.id_to_node_map[node_id]

        ## if node is already visited, ignore it
        if hashable_state(node.state) in Node.visited_states:
            continue

        ## marking new state as visited
        Node.visited_states.add(node_state_string)
        num_expanded_nodes += 1
        
        # 5. If the node is the destination, return the path
        if node.h_val == 0:
            print(f'Number of expanded nodes in A*: {num_expanded_nodes}')
            return node
        # 6. Else, add the children to the queue
        node.check_all_next_states(Node.n, min_heap)
    print(f'Number of expanded nodes in A*: {num_expanded_nodes}')
    # 10. If the queue is empty, return failure
    return None


class MinHeapBFS:
    def __init__(self):
        self.heap = []

    def insert_node(self, node):
        # we push tuple of (objective function, str of state) into heap
        heappush(self.heap, (node.g_val, hashable_state(node.state), node.id))

    def pop_min(self):
        return heappop(self.heap)


def BFS(source):
    # A* search
    # 1. Create a priority queue
    min_heap = MinHeapBFS()
    # 2. Create source node and add it to the queue
    source_node = Node(source, 0, None, 0, None)
    min_heap.insert_node(source_node)
    # 3. While queue is not empty
    num_expanded_nodes = 0
    while min_heap.heap:
        # 4. Pop the node with the lowest cost (expanding queue head)
        # min_heap.pop_min()[1] is the string format of state
        popped_tuple = min_heap.pop_min()
        node_state_string = popped_tuple[1]
        node_id = popped_tuple[2]
        node = Node.id_to_node_map[node_id]

        ## if node is already visited, ignore it
        if hashable_state(node.state) in Node.visited_states:
            continue

        ## marking new state as visited
        Node.visited_states.add(node_state_string)
        num_expanded_nodes += 1

        # 5. If the node is the destination, return the path
        if node.h_val == 0:
            print(f'Number of expanded nodes in BFS: {num_expanded_nodes}')
            return node
        # 6. Else, add the children to the queue
        node.check_all_next_states(Node.n, min_heap)
    print(f'Number of expanded nodes in BFS: {num_expanded_nodes}')
    # 10. If the queue is empty, return failure
    return None

def main():
    n = int(input())
    Node.n = n

    dest_map = {}
    for i in range(n):
        row_list = list(map(int, input().strip().split()))
        for j in range(n):
            dest_map[row_list[j]] = (i, j)
    Node.dest = dest_map

    source = [list(map(int, input().strip().split())) for _ in range(n)]

    # accurately calculate runtime of a_star
    t = Timer(lambda: a_star(source))
    print(f'Runtime of A*: {t.timeit(number=1)}')
    Node.clear()
    t = Timer(lambda: BFS(source))
    print(f'Runtime of BFS: {t.timeit(number=1)}')


if __name__ == '__main__':
    main()
