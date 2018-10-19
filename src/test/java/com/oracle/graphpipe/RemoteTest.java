/*
** Copyright © 2018, Oracle and/or its affiliates. All rights reserved.
** Licensed under the Universal Permissive License v 1.0 as shown at http://oss.oracle.com/licenses/upl.
*/
package com.oracle.graphpipe;

import com.oracle.graphpipefb.InferRequest;
import com.oracle.graphpipefb.Request;
import com.oracle.graphpipefb.Tensor;
import com.oracle.graphpipefb.Type;
import junit.framework.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by aditpras on 7/28/18.
 */
public class RemoteTest extends TestCase {
    // Ctor from INDArray.
    // Get back as native flat/multidimensional array.
    public void testBuildRequest() {
        byte[][] input = {{1, 2, 3}, {4, 5, 6}};
        
        List<NativeTensor> inputs = Arrays.asList(NativeTensor.fromArray(input));
        List<String> inputNames = Arrays.asList();
        List<String> outputNames = Arrays.asList();
       
        ByteBuffer req = Remote.BuildRequest("", inputs, inputNames, outputNames);

        Request r = Request.getRootAsRequest(req);
        InferRequest ir = new InferRequest();
        r.req(ir);

        assertEquals(inputNames.size(), ir.inputNamesLength());
        assertEquals(outputNames.size(), ir.outputNamesLength());
        assertEquals(inputs.size(), ir.inputTensorsLength());

        Tensor t = ir.inputTensors(0);
        assertEquals(Type.Int8, t.type());
        assertEquals(2, t.shapeLength());
        assertEquals(2, t.shape(0));
        assertEquals(3, t.shape(1));
        
        List<Integer> data = new ArrayList<>();
        for (int i = 0; i < t.dataLength(); i++) {
            data.add(t.data(i));
        }
        assertEquals(Arrays.asList(1, 2, 3, 4, 5, 6), data);
    }
   
    // TODO: Currently requires Aditya's modified RemoteModelWithGraphPipe.ipynb
    public void testRemote() throws IOException {
//        float[][][] input = {{{1, 2, 3}, {4, 5, 6}}};
//        NativeTensor nt = Remote.Execute("http://localhost:9000",
//                NativeTensor.fromArray(input));
//
//        INDArray ndArr = nt.toINDArray();
//        assertEquals((1 + 2 + 3) * 2.0, ndArr.getDouble(0, 0, 0));
//        assertEquals((4 + 5 + 6) * 2.0, ndArr.getDouble(0, 1, 0));
//        assertEquals(2, ndArr.length());
    }
}
