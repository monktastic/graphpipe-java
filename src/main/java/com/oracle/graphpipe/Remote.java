package com.oracle.graphpipe;

import com.google.flatbuffers.FlatBufferBuilder;
import com.oracle.graphpipefb.*;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Remote {
    public static ByteBuffer BuildRequest(
            String config, List<NativeTensor> inputs, List<String> inputNames, 
            List<String> outputNames) {
        FlatBufferBuilder b = new FlatBufferBuilder(1024);

        int[] inputNameOffsets = new int[inputNames.size()];
        for (int i = 0; i < inputNames.size(); i++) {
            inputNameOffsets[i] = b.createString(inputNames.get(i));
        }
        int inputNamesOffset = InferRequest.createInputNamesVector(b, inputNameOffsets);

        int[] outputNameOffsets = new int[outputNames.size()];
        for (int i = 0; i < outputNames.size(); i++) {
            outputNameOffsets[i] = b.createString(outputNames.get(i));
        }
        int outputNamesOffset = InferRequest.createOutputNamesVector(b, outputNameOffsets);

        int[] tensorOffsets = new int[inputs.size()];
        for (int i = 0; i < inputs.size(); i++) {
            tensorOffsets[i] = inputs.get(i).Build(b);
        }
        int tensorsOffset = InferRequest.createInputTensorsVector(b, tensorOffsets);

        int configOffset = b.createString(config);
        
        InferRequest.startInferRequest(b);
        InferRequest.addInputNames(b, inputNamesOffset);
        InferRequest.addOutputNames(b, outputNamesOffset);
        InferRequest.addInputTensors(b, tensorsOffset);
        InferRequest.addConfig(b, configOffset);
        
        int inferRequestOffset = InferRequest.endInferRequest(b);
        Request.startRequest(b);
        Request.addReqType(b, Req.InferRequest);
        Request.addReq(b, inferRequestOffset);
        
        int requestOffset = Request.endRequest(b);
        Request.finishRequestBuffer(b, requestOffset);
        return b.dataBuffer();
    }
    
    public static NativeTensor Execute(String uri, NativeTensor input) 
            throws IOException {
        List<NativeTensor> inputs = Collections.singletonList(input);
        List<String> inputNames = Collections.emptyList();
        List<String> outputNames = Collections.emptyList();
        return ExecuteMulti("", uri, inputs, inputNames, outputNames).get(0);
    }

    public static NativeTensor Execute(
            String uri, NativeTensor input, String inputName, String outputName)
            throws IOException {
        List<NativeTensor> inputs = Collections.singletonList(input);
        List<String> inputNames = Collections.singletonList(inputName);
        List<String> outputNames = Collections.singletonList(outputName);
        return ExecuteMulti("", uri, inputs, inputNames, outputNames).get(0);
    }
    
    public static NativeTensor Execute(
            String uri, NativeTensor input, String inputName, String 
            outputName, String config)
            throws IOException {
        List<NativeTensor> inputs = Collections.singletonList(input);
        List<String> inputNames = Collections.singletonList(inputName);
        List<String> outputNames = Collections.singletonList(outputName);
        return ExecuteMulti(config, uri, inputs, inputNames, outputNames).get(0);
    }

    public static List<NativeTensor> ExecuteMulti(
            String config, String uri, List<NativeTensor> inputs,
            List<String> inputNames, List<String> outputNames)
            throws IOException {
        ByteBuffer req = BuildRequest(config, inputs, inputNames, outputNames);
        
        byte[] arr = new byte[req.remaining()];
        req.get(arr);

        CloseableHttpClient httpclient = HttpClients.createDefault();
        HttpPost post = new HttpPost(uri);
        HttpEntity entity = new ByteArrayEntity(arr);
        post.setEntity(entity);
        CloseableHttpResponse response = httpclient.execute(post);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        response.getEntity().writeTo(baos);
        
        byte[] respBytes = baos.toByteArray();
        ByteBuffer respBB = ByteBuffer.wrap(respBytes);
        InferResponse ir = InferResponse.getRootAsInferResponse(respBB);
       
        List<NativeTensor> nTensors = new ArrayList<>(ir.outputTensorsLength());
        for (int i = 0; i < ir.outputTensorsLength(); i++) {
            Tensor t = ir.outputTensors(0);
            nTensors.add(NativeTensor.fromTensor(t));
        }
        return nTensors;
    }
}
