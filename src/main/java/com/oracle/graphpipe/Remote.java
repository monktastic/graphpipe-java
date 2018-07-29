package com.oracle.graphpipe;

import com.google.flatbuffers.FlatBufferBuilder;
import com.oracle.graphpipefb.InferRequest;
import com.oracle.graphpipefb.Req;
import com.oracle.graphpipefb.Request;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Remote {
    public static ByteBuffer BuildRequest(List<NativeTensor> inputs,
                                          List<String> inputNames, List<String> outputNames) {
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

        InferRequest.startInferRequest(b);
        InferRequest.addInputNames(b, inputNamesOffset);
        InferRequest.addOutputNames(b, outputNamesOffset);
        InferRequest.addInputTensors(b, tensorsOffset);
        
        int inferRequestOffset = InferRequest.endInferRequest(b);
        Request.startRequest(b);
        Request.addReqType(b, Req.InferRequest);
        Request.addReq(b, inferRequestOffset);
        
        int requestOffset = Request.endRequest(b);
        Request.finishRequestBuffer(b, requestOffset);
        return b.dataBuffer();
    }
    
    public static NativeTensor Execute(String uri, NativeTensor input, String inputName, String outputName) {
        List<NativeTensor> inputs = Collections.singletonList(input);
        List<String> inputNames = Collections.singletonList(inputName);
        List<String> outputNames = Collections.singletonList(outputName);
        return ExecuteMulti(uri, inputs, inputNames, outputNames).get(0);
    }

    public static List<NativeTensor> ExecuteMulti(String uri, List<NativeTensor> inputs,
                                      List<String> inputNames, List<String> outputNames) {
        ByteBuffer req = BuildRequest(inputs, inputNames, outputNames);
        return new ArrayList<NativeTensor>();
    }
}
