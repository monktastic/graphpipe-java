package com.oracle.graphpipe;

import com.google.flatbuffers.FlatBufferBuilder;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

public class NativeTensor {
    public int Build(FlatBufferBuilder b) {
        return 0;
    }

    static boolean isRectAry(Object ary) {
        if (!ary.getClass().isArray()) {
            return false;
        }
        int nDims = 1 + ary.getClass().getName().lastIndexOf('[');

        List<Integer> sizes = new ArrayList<Integer>(nDims);
        for (int i = 0; i < nDims; i++) {
            sizes.add(-1);
        }

        return _isRectAry(ary, 0, sizes);
    }

    static boolean _isRectAry(Object ary, int dim, List<Integer> sizes) {
        if (sizes.get(dim) == -1) {
            sizes.set(dim, Array.getLength(ary));
        } else if (sizes.get(dim) != Array.getLength(ary)) {
            return false;
        }

        if (dim == sizes.size() - 1) {
            return true;
        }

        for (int i = 0; i < sizes.get(dim); i++) {
            if (!_isRectAry(Array.get(ary, i), dim + 1, sizes)) {
                return false;
            }
        }
        return true;
    }
}
