package com.oracle.graphpipe;

import com.oracle.graphpipefb.InferRequest;
import com.oracle.graphpipefb.InferResponse;
import com.oracle.graphpipefb.Tensor;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;


public class Main {

    public static void main(String[] args) throws IOException {
        byte[][][] input = {{{1, 2, 3}, {4, 5, 6}}};
        
        List<NativeTensor> inputs = Arrays.asList(new NativeTensor(input));
        List<String> inputNames = Arrays.asList();
        List<String> outputNames = Arrays.asList();
        ByteBuffer req = Remote.BuildRequest(inputs, inputNames, outputNames);
                
        byte[] arr = new byte[req.remaining()];
        req.get(arr);
        
        CloseableHttpClient httpclient = HttpClients.createDefault();
        HttpPost post = new HttpPost("http://localhost:9000");
        HttpEntity entity = new ByteArrayEntity(arr);
        post.setEntity(entity);
        CloseableHttpResponse response = httpclient.execute(post);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        response.getEntity().writeTo(baos);
        byte[] respBytes = baos.toByteArray();
        ByteBuffer respBB = ByteBuffer.wrap(respBytes);
        InferResponse ir = InferResponse.getRootAsInferResponse(respBB);
        Tensor t = ir.outputTensors(0);
        System.out.println("Shape length " + t.shapeLength());
    }

}
