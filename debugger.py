import numpy as np
import os

class Debugger:
    def __init__(self, activated):
        self.activated = activated

    def clear_terminal(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def start_iteration(self, iteration):
        if self.activated:
            print("Iteration start: {}".format(iteration))
            print("")

    def start_sequence(self, current_sequence, sequence_length):
        if self.activated:
            print("Current sequence {}/{}".format(current_sequence, sequence_length))

    def print_vector(self, label, vector):
        if self.activated:
            print("{}\t".format(label), end="")
            for i in range(vector.shape[1]):
                print("| {:.5f} ".format(vector[0][i]), end="")
            print("|")
            print("")

    def print_matrix(self, label, matrix):
        if self.activated:
            print("{}\t".format(label))
            for i in range(matrix.shape[0]):
                print("\t\t", end="")
                for j in range(matrix.shape[1]):
                    print("| {:.5f} ".format(matrix[i][j]), end="")
                print("|")
            print("")

    def print_gate_info(self, gate_name, Wh, Wx, bias, a, activation_function, output):
        if self.activated:
            print("")
            print(gate_name)
            print("==============")
            self.print_matrix("Wh:     ", Wh)
            self.print_matrix("Wx:     ", Wx)
            self.print_vector("B:     ", np.array([bias]))

            print("A = prev_h.dot(Wh) + x.dot(Wx) + b")
            self.print_vector("A:     ", a)
            print("Activation: ", activation_function)
            self.print_vector("Output:  ", output)
            self.wait_on_key()

    def step_forward_output(self, next_c, next_h):
        print("Step forward output")
        print("==============")
        print("next_c = f * prev_c + i * g")
        self.print_vector("next_c:", next_c)
        print("next_h = o * tanh(next_c))")
        self.print_vector("next_h:", next_h)
        self.wait_on_key()

    def step_forward(self, x, prev_h, prev_c):
        if self.activated:
            self.print_vector("x:     ", x)
            self.print_vector("prev_h:", prev_h)
            self.print_vector("prev_c:", prev_c)
            self.wait_on_key()

    def wait_on_key(self):
        if self.activated:
            input("Press Enter to continue...")
