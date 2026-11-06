package br.com.ecommerce.ml;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.ndarray.NDManager;
import org.junit.jupiter.api.Test;
import org.tensorflow.Result;
import org.tensorflow.Session;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

class EcommerceRecommenderTest {

    @Test
    void predictsOneScorePerUserProductPair() {
        float[][] features = {
                {25.0f, 150.5f, 299.9f, 5.0f},
                {40.0f, 800.0f, 45.0f, 2.0f}
        };

        try (NDManager manager = NDManager.newBaseManager();
             EcommerceRecommender.TensorFlowGraph model = EcommerceRecommender.createModel();
             Session session = new Session(model.graph());
             TFloat32 input = TFloat32.tensorOf(StdArrays.ndCopyOf(features));
             Result result = session.runner()
                     .feed(model.input(), input)
                     .fetch(model.output())
                     .run()) {
            float[][] values = StdArrays.array2dCopyOf((TFloat32) result.get(0));
            assertEquals(2, values.length);
            for (float[] row : values) {
                assertEquals(1, row.length);
                assertTrue(Float.isFinite(row[0]));
                assertTrue(row[0] >= 0.0f && row[0] <= 1.0f);
            }
            assertEquals(2, manager.create(values).getShape().get(0));
        }
    }
}
