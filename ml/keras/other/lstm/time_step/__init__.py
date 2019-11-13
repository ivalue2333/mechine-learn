"""
    return_sequences
        return_sequences=false，那么只返回最后一个time step 的hidden unit的输出
        如果设置为True，则返回每一个time step 的hidden unit的输出

    return_state，
        The LSTM hidden state output for the last time step.
        The LSTM hidden state output for the last time step (again).
        The LSTM cell state for the last time step.


    Point !!!
        That return sequences return the hidden state output for each input time step.
        That return state returns the hidden state output and cell state for the last input time step.
        That return sequences and return state can be used at the same time.

        return sequences 为每一个time step返回hidden state output
        return state 为最后一次的time step返回 hidden state output 和 cell state
        他们两个可以同时使用

        LSTM 的网络结构中，直接根据当前 input 数据，得到的输出称为 hidden state。
        还有一种数据是不仅仅依赖于当前输入数据，而是一种伴随整个网络过程中用来记忆，遗忘，
        选择并最终影响 hidden state 结果的东西，称为 cell state。 cell state 就是实现 long short memory 的关键。
        简单说就是下一个lstm（当然这说的是复杂lstm网络）的cell state是由上一个lstm的cell state来初始化的。

        Generally, we do not need to access the cell state unless we are developing sophisticated models
        where subsequent layers may need to have their cell state initialized with t
        he final cell state of another layer, such as in an encoder-decoder model.

"""