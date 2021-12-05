import random
from pickle import dump, load
import numpy as np
from activations import activation
from debugger import Debugger

class LTSM:
    def __init__(self, input_dimension, sequence_length=60, hidden_dim=250, learning_rate=1e-2, debug=False) -> None:
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.debug = debug
        self.debugger = Debugger(debug)

        # model parameters
        self.Wx = np.random.randn(input_dimension, 4 * hidden_dim) / np.sqrt(4 * hidden_dim)  # input to hidden
        self.Wh = np.random.randn(hidden_dim, 4 * hidden_dim) / np.sqrt(4 * hidden_dim)  # hidden to hidden
        self.b = np.zeros(4 * hidden_dim)  # hidden bias
        self.Why = np.random.randn(input_dimension, hidden_dim) / np.sqrt(hidden_dim)
        self.by = np.zeros(input_dimension)
        self.prev_h = np.zeros((1, hidden_dim))  # reset LSTM memory

    def fit(self, input_data, ground_truth, iterations=2000):
        loss_list = [-np.log(1.0 / self.input_dimension)]
        smooth_loss = loss_list.copy()
        for i in range(1, iterations + 1):
            self.debugger.start_iteration(i)
            index = random.randint(0, input_data.shape[0] - 1)
            data = input_data[index]
            targets = ground_truth[index]
            predict, h_states, h_cache = self.forward(data)
            loss = self.loss_function(predict, targets)
            if i % 100 == 0:
                print('{}/{}: {}'.format(i, iterations, loss))
            loss_list.append(loss)
            smooth_loss.append(smooth_loss[-1] * 0.999 + loss_list[-1] * 0.001)

            # Backward pass
            self.backpropagation(predict, targets, h_states, h_cache)
            self.prev_h = h_states[-1][None]

        return loss_list, smooth_loss

    def predict(self, input_data):
        data = []
        for i in range(input_data.shape[0]):
            input = input_data[i]
            predict, h_states, h_cache = self.forward(input)
            data.append(predict)

        return np.array(data)

    @staticmethod
    def loss_function(predicted, ground_truth):
        diff = pow(predicted - ground_truth, 2)
        diff_sum = np.sum(diff)
        return diff_sum / len(predicted)

    def forward(self, x):
        """
        Inputs:
        - x: Input data of shape (seq_length, input_dim)
        - h0: Initial hidden state of shape (1, hidden_dim)
        - Wx: Weights for input-to-hidden connections, of shape (input_dim, 4*hidden_dim)
        - Wh: Weights for hidden-to-hidden connections, of shape (hidden_dim, 4*hidden_dim)
        - b: Biases of shape (4*hidden_dim)
        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (seq_length, hidden_dim)
        - cache: Values needed for the backward pass.
        """
        h = None
        cache = []
        prev_c = np.zeros_like(self.prev_h)
        prev_h = np.zeros_like(self.prev_h)
        for i in range(self.sequence_length):  # 0 to seq_length-1
            self.debugger.start_sequence(i + 1, self.sequence_length)
            next_h, next_c, next_cache = self.step_forward(x[i][None], prev_h, prev_c, self.Wx, self.Wh, self.b)
            self.prev_h = next_h
            prev_c = next_c
            prev_h = next_h
            cache.append(next_cache)
            if i > 0:
                h = np.append(h, next_h, axis=0)
            else:
                h = next_h

        scores = np.dot(h, self.Why.T) + self.by  # (seq_length, input_dim)
        predict = activation('sigmoid', scores)

        return predict, h, cache

    def backpropagation(self, predict, targets, h_states, cache):
        """
        Backward pass for an LSTM over an entire sequence of data.]
        Inputs:
        - dh: Upstream gradients of hidden states, of shape (N, T, H) (seq_length, hidden_dim)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dscores = (predict - targets) * (predict * (1.0 - predict))
        # dscores = (np.exp(-dscores))/(np.power(1 + np.exp(-dscores), 2))
        dWhy = dscores.T.dot(h_states)
        dby = np.sum(dscores, axis=0)
        dh_states = dscores.dot(self.Why)
        dx, dh0, dWx, dWh, db = None, None, None, None, None
        N, H = dh_states.shape
        dh_prev = 0
        dc_prev = 0
        for i in reversed(range(N)):
            dx_step, dh0_step, dc_step, dWx_step, dWh_step, db_step = self.step_backward(dh_states[i][None] + dh_prev, dc_prev,
                                                                                         cache[i])
            dh_prev = dh0_step
            dc_prev = dc_step
            if i == N - 1:
                dx = dx_step
                dWx = dWx_step
                dWh = dWh_step
                db = db_step
            else:
                dx = np.append(dx_step, dx, axis=0)
                dWx += dWx_step
                dWh += dWh_step
                db += db_step
        dh0 = dh0_step

        ### Gradient update ###
        self.Wx -= self.learning_rate * dWx * 0.5
        self.Wh -= self.learning_rate * dWh * 0.5
        self.Why -= self.learning_rate * dWhy * 0.5
        self.b -= self.learning_rate * db * 0.5
        self.by -= self.learning_rate * dby * 0.5

        return dx, dh0, dWx, dWh, db

    def step_forward(self, x, prev_h, prev_c, Wx, Wh, b):
        """
        Forward pass for a single timestep of an LSTM.
        The input data has dimension D, the hidden state has dimension H, and we use
        a minibatch size of N.
        Inputs:
        - x: Input data, of shape (1, input_dim)
        - prev_h: Previous hidden state, of shape (1, hidden_dim)
        - prev_c: previous cell state, of shape (1, hidden_dim)
        - Wx: Input-to-hidden weights, of shape (input_dim, 4*hidden_dim)
        - Wh: Hidden-to-hidden weights, of shape (hidden_dim, 4*hidden_dim)
        - b: Biases, of shape (4*hidden_dim)
        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        _, H = prev_h.shape
        self.debugger.step_forward(x, prev_h, prev_c)
        a = prev_h.dot(Wh) + x.dot(Wx) + b  # (1, 4*hidden_dim)
        i = activation('sigmoid', a[:, 0:H])
        self.debugger.print_gate_info("Input gate", Wh[:, 0:H], Wx[:, 0:H], b[0:H], a[:, 0:H], 'sigmoid', i)
        f = activation('sigmoid', a[:, H:2 * H])
        self.debugger.print_gate_info("Forget gate", Wh[:, H:2 * H], Wx[:, H:2 * H], b[H:2 * H], a[:, H:2 * H],
                                      'sigmoid', f)
        o = activation('sigmoid', a[:, 2 * H:3 * H])
        self.debugger.print_gate_info("Output gate", Wh[:, 2 * H:3 * H], Wx[:, 2 * H:3 * H], b[2 * H:3 * H],
                                      a[:, 2 * H:3 * H], 'sigmoid', o)
        g = activation('tanh', a[:, 3 * H:4 * H])  # (1, hidden_dim)
        self.debugger.print_gate_info("Candidate gate", Wh[:, 3 * H:4 * H], Wx[:, 3 * H:4 * H], b[3 * H:4 * H],
                                      a[:, 3 * H:4 * H], 'tanh', g)
        next_c = f * prev_c + i * g  # (1, hidden_dim)
        next_h = o * (activation('tanh', next_c))  # (1, hidden_dim)
        self.debugger.step_forward_output(next_c, next_h)
        cache = x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c
        return next_h, next_c, cache

    def step_backward(self, dnext_h, dnext_c, cache):
        """
        Backward pass for a single timestep of an LSTM.
        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c = cache
        _, H = dnext_h.shape
        d1 = o * (1 - np.tanh(next_c) ** 2) * dnext_h + dnext_c
        dprev_c = f * d1
        dop = np.tanh(next_c) * dnext_h
        dfp = prev_c * d1
        dip = g * d1
        dgp = i * d1
        do = o * (1 - o) * dop
        df = f * (1 - f) * dfp
        di = i * (1 - i) * dip
        dg = (1 - g ** 2) * dgp
        da = np.concatenate((di, df, do, dg), axis=1)
        db = np.sum(da, axis=0)
        dx = da.dot(Wx.T)
        dprev_h = da.dot(Wh.T)
        dWx = x.T.dot(da)
        dWh = prev_h.T.dot(da)
        return dx, dprev_h, dprev_c, dWx, dWh, db

    def save(self, filename):
        a_file = open(filename, "wb")
        dump({'Wx': self.Wx, 'Wh': self.Wh, 'b': self.b, 'Why': self.Why, 'by': self.by, 'prev_h': self.prev_h}, a_file)
        a_file.close()

    def load(self, filename):
        a_file = open(filename, "rb")
        output = load(a_file)
        self.Wx = output['Wx']
        self.Wh = output['Wh']
        self.b = output['b']
        self.Why = output['Why']
        self.by = output['by']
        self.prev_h = output['prev_h']
        a_file.close()
