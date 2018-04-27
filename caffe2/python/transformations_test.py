# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from caffe2.python.transformations import (
    addNNPACK,
    fuseNNPACKConvRelu,
    fuseConvRelu,
    fuseAveragePoolRelu,
    fuseMaxPoolRelu,
    fuseSumRelu,
    fuseConvBN,
)
from caffe2.python import core, workspace, test_util


class TestTransformations(test_util.TestCase):
    def test_addNNPACK(self):
        net = core.Net("net")
        net.Conv(
            ["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW"
        )
        net.Relu(["Y"], ["Y2"])
        addNNPACK(net)
        assert (net.Proto().op[0].engine == "NNPACK")


    def test_fuseNNPACKConvRelu(self):
        net = core.Net("net")
        net.Conv(
            ["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW"
        )
        net.Relu(["Y"], ["Y2"])
        addNNPACK(net) # get the NNPACK engine
        assert (net.Proto().op[0].engine == "NNPACK")
        fuseNNPACKConvRelu(net)
        assert (len(net.Proto().op) == 1)
        has_activation_arg = False
        for arg in net.Proto().op[0].arg:
            if arg.name == "activation":
                assert (arg.s == "Relu")
                has_activation_arg = True
        assert has_activation_arg

    def test_noFuseNNPACKConvRelu(self):
        net = core.Net("net")
        net.Conv(
            ["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW"
        )
        net.Relu(["Y"], ["Y2"])
        net.Relu(["Y"], ["Y3"])
        addNNPACK(net) # get the NNPACK engine
        assert (net.Proto().op[0].engine == "NNPACK")
        fuseNNPACKConvRelu(net)
        assert (len(net.Proto().op) == 3)
        has_activation_arg = False
        for arg in net.Proto().op[0].arg:
            if arg.name == "activation" and arg.s == "Relu":
                has_activation_arg = True
        assert not has_activation_arg

    def test_fuseNNPACKConvReluNoInplace(self):
        net = core.Net("net")
        net.Conv(
            ["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW"
        )
        net.Relu(["Y"], ["X"])
        addNNPACK(net) # get the NNPACK engine
        assert (net.Proto().op[0].engine == "NNPACK")
        fuseNNPACKConvRelu(net)
        assert (len(net.Proto().op) == 1)
        has_activation_arg = False
        for arg in net.Proto().op[0].arg:
            if arg.name == "activation":
                assert (arg.s == "Relu")
                has_activation_arg = True
        assert has_activation_arg
        assert net.Proto().op[0].output[0] != net.Proto().op[0].input[0]

    def test_fuseNNPACKConvReluInplaceRelu(self):
        net = core.Net("net")
        net.Conv(
            ["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW"
        )
        net.Relu(["Y"], ["Y"])
        addNNPACK(net) # get the NNPACK engine
        assert (net.Proto().op[0].engine == "NNPACK")
        fuseNNPACKConvRelu(net)
        assert (len(net.Proto().op) == 1)
        has_activation_arg = False
        for arg in net.Proto().op[0].arg:
            if arg.name == "activation":
                assert (arg.s == "Relu")
                has_activation_arg = True
        assert has_activation_arg
        assert net.Proto().op[0].output[0] != net.Proto().op[0].input[0]

    def test_fuseConvRelu(self):
        net = core.Net("net")
        net.Conv(
            ["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW"
        )
        net.Relu(["Y"], ["Y2"])
        fuseConvRelu(net)
        assert (len(net.Proto().op) == 1)

        # Now we test a situtation where fusion is not possible
        net2 = core.Net("net2")
        net2.Conv(
            ["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=3, order="NCHW"
        )
        net2.Relu(["Y"], ["Y2"])
        net2.AveragePool(["Y"], ["Y3"], kernel=3)
        fuseConvRelu(net2)
        # There won't be any fusion in this case
        assert (len(net2.Proto().op) == 3)

    def test_fuseAveragePoolRelu(self):
        net = core.Net("net")
        net.AveragePool(["X"], ["Y"], kernel=3, order="NCHW")
        net.Relu(["Y"], ["Y2"])
        fuseAveragePoolRelu(net)
        assert (len(net.Proto().op) == 1)

        # Now we test a situtation where fusion is not possible
        net2 = core.Net("net2")
        net2.AveragePool(["X"], ["Y"], kernel=3, order="NCHW")
        net2.Relu(["Y"], ["Y2"])
        net2.AveragePool(["Y"], ["Y3"], kernel=3)
        fuseAveragePoolRelu(net2)
        # There won't be any fusion in this case
        assert (len(net2.Proto().op) == 3)

    def test_fuseMaxPoolRelu(self):
        net = core.Net("net")
        net.MaxPool(["X"], ["Y"], kernel=3, order="NCHW")
        net.Relu(["Y"], ["Y2"])
        fuseMaxPoolRelu(net)
        assert (len(net.Proto().op) == 1)

        # Now we test a situtation where fusion is not possible
        net2 = core.Net("net2")
        net2.MaxPool(["X"], ["Y"], kernel=3, order="NCHW")
        net2.Relu(["Y"], ["Y2"])
        net2.MaxPool(["Y"], ["Y3"], kernel=3)
        fuseMaxPoolRelu(net2)
        # There won't be any fusion in this case
        assert (len(net2.Proto().op) == 3)

    def test_fuseSumRelu(self):
        net = core.Net("net")
        net.Sum(["X"], ["Y"], kernel=3, order="NCHW")
        net.Relu(["Y"], ["Y2"])
        fuseSumRelu(net)
        assert (len(net.Proto().op) == 1)

        # Now we test a situtation where fusion is not possible
        net2 = core.Net("net2")
        net2.Sum(["X"], ["Y"], kernel=3, order="NCHW")
        net2.Relu(["Y"], ["Y2"])
        net2.Sum(["Y"], ["Y3"], kernel=3)
        fuseSumRelu(net2)
        # There won't be any fusion in this case
        assert (len(net2.Proto().op) == 3)

    @given(
        size=st.integers(7, 10),
        input_channels=st.integers(1, 10),
        seed=st.integers(0, 65535),
        order=st.sampled_from(["NCHW", "NHWC"]),
        epsilon=st.floats(min_value=1e-5, max_value=1e-2)
    )
    def test_fuseConvBN(self, size, input_channels, seed, order, epsilon):
        net = core.Net("net")
        c = input_channels
        h = size
        w = size
        k = 3
        net.Conv(["X", "w", "b"], ["Y"], stride=1, pad=0, kernel=k, order=order)
        net.SpatialBN(
            ["Y", "scale", "bias", "mean", "var"], ["Y2"],
            is_test=True,
            order=order,
            epsilon=epsilon
        )

        np.random.seed(seed)
        if order == "NCHW":
            workspace.FeedBlob(
                "X",
                np.random.rand(1, c, h, w).astype(np.float32)
            )
            workspace.FeedBlob(
                "w",
                np.random.rand(c, c, k, k).astype(np.float32)
            )
        else:
            workspace.FeedBlob(
                "X",
                np.random.rand(1, h, w, c).astype(np.float32)
            )
            workspace.FeedBlob(
                "w",
                np.random.rand(c, k, k, c).astype(np.float32)
            )
        workspace.FeedBlob("b", np.random.rand(c).astype(np.float32))
        workspace.FeedBlob("scale", np.random.rand(c).astype(np.float32))
        workspace.FeedBlob("bias", np.random.rand(c).astype(np.float32))
        workspace.FeedBlob("mean", np.random.rand(c).astype(np.float32))
        workspace.FeedBlob("var", np.random.rand(c).astype(np.float32))
        workspace.RunNetOnce(net)
        preTransformOutput = workspace.FetchBlob("Y2")
        fuseConvBN(net)

        # Ensure fusion
        print(net.Proto().op)
        assert (len(net.Proto().op) == 1)
        workspace.RunNetOnce(net)
        postTransformOutput = workspace.FetchBlob("Y2")
        # Check that there is no numerical difference
        assert (np.allclose(preTransformOutput, postTransformOutput, rtol=1e-05, atol=1e-08))
