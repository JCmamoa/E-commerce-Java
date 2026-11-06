package br.com.ecommerce.ml;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import org.tensorflow.Result;
import org.tensorflow.Session;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

public final class TensorOperations {

    private TensorOperations() {}

    public static void main(String[] args) {
        float[][] userProductFeatures = {
                {25.0f, 150.5f, 299.9f, 5.0f},
                {40.0f, 800.0f, 45.0f, 2.0f}
        };

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray inputTensor = manager.create(userProductFeatures);
            System.out.println("Formato do tensor de entrada: " + inputTensor.getShape());
            float[][] features = toRows(inputTensor.toFloatArray(),
                    (int) inputTensor.getShape().get(0),
                    (int) inputTensor.getShape().get(1));

            try (EcommerceRecommender.TensorFlowGraph model = EcommerceRecommender.createModel();
                 Session session = new Session(model.graph());
                 TFloat32 input = TFloat32.tensorOf(StdArrays.ndCopyOf(features));
                 Result result = session.runner()
                         .feed(model.input(), input)
                         .fetch(model.output())
                         .run()) {
                float[][] output = StdArrays.array2dCopyOf((TFloat32) result.get(0));
                NDArray predictions = manager.create(output);
                System.out.println("Formato do tensor de saída: " + predictions.getShape());
                System.out.println("Scores do modelo (pesos de demonstração, não treinado):");

                float[] scores = predictions.toFloatArray();
                for (int i = 0; i < scores.length; i++) {
                    System.out.printf("Interação %d: %.2f%%%n", i + 1, scores[i] * 100);
                }
            }
        }
    }

    private static float[][] toRows(float[] values, int rows, int columns) {
        float[][] matrix = new float[rows][columns];
        for (int row = 0; row < rows; row++) {
            System.arraycopy(values, row * columns, matrix[row], 0, columns);
        }
        return matrix;
    }
}
