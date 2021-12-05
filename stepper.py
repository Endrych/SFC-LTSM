import numpy as np
import os


class Stepper:
    def __init__(self, activated):
        self.activated = activated

    def start_iteration(self, iteration):
        if self.activated:
            print("Iteration start: {}".format(iteration))
            print("")

    def start_sequence(self, current_sequence, sequence_length):
        if self.activated:
            print("")
            print("Current sequence {}/{}".format(current_sequence, sequence_length))

    def print_vector(self, label, vector):
        if self.activated:
            print("{}\t".format(label), end="")
            for i in range(vector.shape[1]):
                print("| {:.7f} ".format(vector[0][i]), end="")
            print("|")
            print("")

    def print_matrix(self, label, matrix):
        if self.activated:
            print("{}\t".format(label))
            for i in range(matrix.shape[0]):
                print("\t\t", end="")
                for j in range(matrix.shape[1]):
                    print("| {:.7f} ".format(matrix[i][j]), end="")
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
        if self.activated:
            print("")
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

    def forward_output(self, h, WhyT, by, scores, predict):
        if self.activated:
            print("")
            print("Forward output")
            print("==============")
            self.print_matrix("h:", h)
            self.print_matrix("WhyT:", WhyT)
            self.print_vector("by", np.array([by]))
            print("scores = np.dot(h, self.Why.T) + self.by")
            self.print_vector("scores:", scores.T)
            print("predict = sigmoid(scores)")
            self.print_vector("predict:", predict.T)
            self.wait_on_key()

    def backpropagation_start(self, predicted, ground_truth):
        if self.activated:
            print("")
            print("Backpropagation")
            print("==============")
            self.print_vector("Predicted:    ", predicted)
            self.print_vector("Ground truth: ", ground_truth)
            self.wait_on_key()

    def gradient_update(self, label, old, delta, new, is_vector=True):
        if self.activated:
            print("")
            print(label)
            print("==============")
            if is_vector:
                self.print_vector("Old: ", old)
                self.print_vector("Delta: ", delta)
                self.print_vector("New: ", new)
            else:
                self.print_matrix("Old: ", old)
                self.print_matrix("Delta: ", delta)
                self.print_matrix("New: ", new)
            self.wait_on_key()

    def wait_on_key(self):
        if self.activated:
            input("Press Enter to continue...")
