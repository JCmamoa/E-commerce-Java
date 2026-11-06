package br.com.ecommerce.ml;

import java.util.Random;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.types.TFloat32;

public final class EcommerceRecommender {

    public static final int FEATURE_COUNT = 4;
    private static final int HIDDEN_LAYER_1_UNITS = 128;
    private static final int HIDDEN_LAYER_2_UNITS = 64;

    private EcommerceRecommender() {}

    public static TensorFlowGraph createModel() {
        Graph graph = new Graph();
        Ops tf = Ops.create(graph);

        Operand<TFloat32> input = tf.withName("features")
                .placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, FEATURE_COUNT)));

        Operand<TFloat32> hidden1 = tf.nn.relu(tf.math.add(
                tf.linalg.matMul(input, tf.constant(weights(FEATURE_COUNT, HIDDEN_LAYER_1_UNITS))),
                tf.constant(new float[HIDDEN_LAYER_1_UNITS])));
        Operand<TFloat32> hidden2 = tf.nn.relu(tf.math.add(
                tf.linalg.matMul(hidden1, tf.constant(weights(
                        HIDDEN_LAYER_1_UNITS, HIDDEN_LAYER_2_UNITS))),
                tf.constant(new float[HIDDEN_LAYER_2_UNITS])));
        Operand<TFloat32> probability = tf.math.sigmoid(tf.math.add(
                tf.linalg.matMul(hidden2, tf.constant(weights(HIDDEN_LAYER_2_UNITS, 1))),
                tf.constant(new float[1])));

        return new TensorFlowGraph(graph, input.asOutput(), probability.asOutput());
    }

    private static float[][] weights(int inputUnits, int outputUnits) {
        float[][] values = new float[inputUnits][outputUnits];
        Random random = new Random(42L + inputUnits * 31L + outputUnits);
        float limit = (float) Math.sqrt(6.0 / (inputUnits + outputUnits));
        for (int row = 0; row < inputUnits; row++) {
            for (int column = 0; column < outputUnits; column++) {
                values[row][column] = (random.nextFloat() * 2.0f - 1.0f) * limit;
            }
        }
        return values;
    }

    public static final class TensorFlowGraph implements AutoCloseable {
        private final Graph graph;
        private final Output<TFloat32> input;
        private final Output<TFloat32> output;

        private TensorFlowGraph(Graph graph, Output<TFloat32> input, Output<TFloat32> output) {
            this.graph = graph;
            this.input = input;
            this.output = output;
        }

        public Graph graph() {
            return graph;
        }

        public Output<TFloat32> input() {
            return input;
        }

        public Output<TFloat32> output() {
            return output;
        }

        @Override
        public void close() {
            graph.close();
        }
    }
}
