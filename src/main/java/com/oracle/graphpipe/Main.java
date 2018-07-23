package com.oracle.graphpipe;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;


public class Main {

    public static void main(String[] args) {
        List<NativeTensor> inputs = Arrays.asList(new NativeTensor(), new NativeTensor());
        List<String> inputNames = Arrays.asList("foo", "bar");
        List<String> outputNames = Arrays.asList("foo", "bar", "baz");
        ByteBuffer req = Remote.BuildRequest(inputs, inputNames, outputNames);
        System.out.println(req.array());
    }

}
