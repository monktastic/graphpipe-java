package com.oracle.graphpipe;

import com.oracle.graphpipefb.InferRequest;
import com.oracle.graphpipefb.Request;
import com.oracle.graphpipefb.Tensor;
import com.oracle.graphpipefb.Type;
import junit.framework.TestCase;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by aditpras on 7/28/18.
 */
public class RemoteTest extends TestCase {
    
    public void testBuildRequest() {
        byte[][] input = {{1, 2, 3}, {4, 5, 6}};
        
        List<NativeTensor> inputs = Arrays.asList(new NativeTensor(input));
        List<String> inputNames = Arrays.asList();
        List<String> outputNames = Arrays.asList();
       
        ByteBuffer req = Remote.BuildRequest(inputs, inputNames, outputNames);

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
}