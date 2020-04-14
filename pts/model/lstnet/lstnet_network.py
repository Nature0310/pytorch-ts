from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTNetBase(nn.module):
    def __init__(
        self,
        num_series: int,
        channels: int,
        kernel_size: int,
        rnn_cell_type: str,
        rnn_num_layers: int,
        skip_rnn_cell_type: str,
        skip_rnn_num_layers: int,
        skip_size: int,
        ar_window: int,
        context_length: int,
        horizon: Optional[int],
        prediction_length: Optional[int],
        dropout_rate: float,
        output_activation: Optional[str],
        scaling: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_series = num_series
        self.channels = channels
        assert (
            channels % skip_size == 0
        ), "number of conv1d `channels` must be divisible by the `skip_size`"
        self.skip_size = skip_size
        assert ar_window > 0, "auto-regressive window must be a positive integer"
        self.ar_window = ar_window
        assert not ((horizon is None)) == (
            prediction_length is None
        ), "Exactly one of `horizon` and `prediction_length` must be set at a time"
        assert horizon is None or horizon > 0, "`horizon` must be greater than zero"
        assert (
            prediction_length is None or prediction_length > 0
        ), "`prediction_length` must be greater than zero"
        self.prediction_length = prediction_length
        self.horizon = horizon
        assert context_length > 0, "`context_length` must be greater than zero"
        self.context_length = context_length
        if output_activation is not None:
            assert output_activation in [
                "sigmoid",
                "tanh",
            ], "`output_activation` must be either 'sigmiod' or 'tanh' "
        self.output_activation = output_activation
        assert rnn_cell_type in [
            "GRU",
            "LSTM",
        ], "`rnn_cell_type` must be either 'GRU' or 'LSTM' "
        assert skip_rnn_cell_type in [
            "GRU",
            "LSTM",
        ], "`skip_rnn_cell_type` must be either 'GRU' or 'LSTM' "
        self.conv_out = context_length - kernel_size + 1
        conv_skip = self.conv_out // skip_size
        assert conv_skip > 0, (
            "conv1d output size must be greater than or equal to `skip_size`\n"
            "Choose a smaller `kernel_size` or bigger `context_length`"
        )
        self.channel_skip_count = conv_skip * skip_size
        self.skip_rnn_c_dim = channels * skip_size

        self.cnn = nn.Conv1d(
            in_channels=num_series, out_channels=channels, kernel_size=kernel_size,
        )
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_cell_type]
        self.rnn = rnn(
            input_size=input_size, ## ??
            hidden_size=channels,
            num_layers=rnn_num_layers,
            dropout=dropout_rate,
            batch_first=False,
        )

        self.dropout = nn.Dropout(p=dropout_rate)

        skip_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[skip_rnn_cell_type]
        self.skip_rnn = skip_rnn(
            input_size=input_size, ## ??
            hidden_size=channels,
            num_layers=skip_rnn_num_layers,
            dropout=dropout_rate,
            batch_first=False,
        )

        self.fc = nn.Linear()


class LSTNetTrain(LSTNetBase):
    pass


class LSTNetPredict(LSTNetBase):
    pass
