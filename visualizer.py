"""
This module creates visualizers.

Author: wangning(wangning45@baidu.com)
Date  : 2022/12/30 15:57 
"""
import paddle.static
from visualdl import LogWriter
from collections import defaultdict


class Visualizer(object):
    """ Wraps a visualdl for tasks use.
    """

    def __init__(self,
                 log_dir,
                 name=""):
        self.writer = LogWriter(logdir=log_dir)
        self.name = name

        self.tag_step = defaultdict(int)

    def update_hparams(self,
                       args):
        """
        Add hyper-parameters information.
        Args:
            args: Arg parser of task settings.

        Returns:
            None
        """
        args_dict = vars(args)
        self.writer.add_hparams(
            hparams_dict=args_dict,
            metrics_list=['none']
        )

    def update_scalars(self,
                       tag_value,
                       step):
        """
        Update multiple tags at one time.
        Args:
            tag_value: {tag: value} dict for multiple updates
            step: update interval instead of absolute steps

        Returns:
            None
        """
        for tag, value in tag_value.items():
            self.writer.add_scalar(tag=tag, step=self.tag_step[tag], value=value)
            self.tag_step[tag] += step

    def update_model(self,
                     model,
                     input_shapes):
        """
        Add model structure in visualdl.
        Args:
            model: network to display
            input_shapes: shapes of all inputs, list

        Returns:
            None
        """
        input_spec = []
        for s in input_shapes:
            input_spec.append(paddle.static.InputSpec(shape=s, dtype='int64'))
        self.writer.add_graph(
            model=model,
            input_spec=input_spec
        )

    def get_name(self):
        """
        Get descriptor name of visualizer.
        Returns:
            instance name
        """
        return self.name
