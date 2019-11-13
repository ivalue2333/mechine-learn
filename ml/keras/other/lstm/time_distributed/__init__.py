"""
    There are two key points to remember when using the TimeDistributed wrapper layer:
        The input must be (at least) 3D.
        The output will be 3D.
    简单说，输入输出都是3D，需要配置

    TimeDistributed将每一个时序的输出，都做一次Dense，相当于是one-to-one了。

"""