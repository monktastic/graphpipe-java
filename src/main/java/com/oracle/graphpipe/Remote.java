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

        int[] inStrs = new int[inputNames.size()];
        int[] outStrs = new int[outputNames.size()];

        for(int i = 0; i < inStrs.length; i++)  {
            inStrs[i] = b.createString(inputNames.get(i));
        }

        for(int i = 0; i < outStrs.length; i++)  {
            outStrs[i] = b.createString(outputNames.get(i));
        }

        InferRequest.startInputNamesVector(b, inStrs.length);
        for(int i = inStrs.length - 1; i >= 0; i--)  {
            b.addOffset(inStrs[i]);
        }

        int inputNamesOffset = b.endVector();

        InferRequest.startOutputNamesVector(b, outStrs.length);
        for(int i = inStrs.length - 1; i >= 0; i--)  {
            b.addOffset(outStrs[i]);
        }

        int outputNamesOffset = b.endVector();

        int[] inputOffsets = new int[inputs.size()];
        for(int i = 0; i < inStrs.length; i++)  {
            inputOffsets[i] = inputs.get(i).Build(b);
        }

        InferRequest.startInputTensorsVector(b, inputOffsets.length);
        for(int i = inputOffsets.length - 1; i >= 0; i--)  {
            b.addOffset(inputOffsets[i]);
        }

        int inputTensors = b.endVector();

        InferRequest.startInferRequest(b);
        InferRequest.addInputNames(b, inputNamesOffset);
        InferRequest.addOutputNames(b, outputNamesOffset);
        InferRequest.addInputTensors(b, inputTensors);
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
