package com.oracle.graphpipe;

import com.google.flatbuffers.FlatBufferBuilder;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NativeTensor {
    /*
    enum Type:uint8 {
     0. Null,
     1. Uint8,
     2. Int8,  // byte
     3. Uint16,
     4. Int16,  // short
     5. Uint32,
     6. Int32,  // int
     7. Uint64,
     8. Int64,  // long
     9. Float16,
    10. Float32, // float
    11. Float64, // double
    12. String,
    */
    
    private int type;
    private List<Integer> shape;
    private List<String>  stringData;
    ByteBuffer data;
    
    abstract static class NumType {
        int type;
        int size;
       
        NumType(int type, int size) {
            this.type = type;
            this.size = size;
        }

        void putAndAdvance(ByteBuffer bb, Object ary) {
            put(bb, ary);
            advance(bb, ary);
        }
        abstract void put(ByteBuffer bb, Object ary);
        void advance(ByteBuffer bb, Object ary) {
            bb.position(bb.position() + Array.getLength(ary) * this.size);
        }
    }
    
    private static NumType byteNt = new NumType(2, 1) {
        void put(ByteBuffer bb, Object ary) {
           bb.put((byte[])ary); 
        }
        void advance(ByteBuffer bb, Object ary) {
            // No-op because put() already advances.
        }
    };
    private static NumType shortNt = new NumType(4, 2) {
        void put(ByteBuffer bb, Object ary) {
            bb.asShortBuffer().put((short[])ary);
        }
    };
    private static NumType intNt = new NumType(6, 4) {
        void put(ByteBuffer bb, Object ary) {
            bb.asIntBuffer().put((int[])ary);
        }
    };
    private static NumType longNt = new NumType(8, 8) {
        void put(ByteBuffer bb, Object ary) {
            bb.asLongBuffer().put((long[])ary);
        }
    };
    private static NumType floatNt = new NumType(10, 8) {
        void put(ByteBuffer bb, Object ary) {
            bb.asFloatBuffer().put((float[])ary);
        }
    };
    private static NumType doubleNt = new NumType(11, 16) {
        void put(ByteBuffer bb, Object ary) {
            bb.asDoubleBuffer().put((double[])ary);
        }
    };

    private static Map<Class<? extends Number>, NumType> typeMap;
    static {
        typeMap = new HashMap<>();
        typeMap.put(Byte.class, byteNt);
        typeMap.put(Short.class, shortNt);
        typeMap.put(Integer.class, intNt);
        typeMap.put(Long.class, longNt);
        typeMap.put(Float.class, floatNt);
        typeMap.put(Double.class, doubleNt);
    }
   
    public int Build(FlatBufferBuilder b) {
        return 0;
    }
   
    private static int getDims(Object ary) {
        return 1 + ary.getClass().getName().lastIndexOf('[');
    }
    
    private static Class<?> getAryType(Object ary) {
        if (!ary.getClass().isArray()) {
            throw new IllegalArgumentException("Not an array");
        }
        int nDims = getDims(ary);
        for (int i = 0; i < nDims; i++) {
            // The final dimension may be empty.
            if (Array.getLength(ary) == 0) {
               throw new IllegalArgumentException("Array is empty");
            }
            ary = Array.get(ary, 0);
        }
        return ary.getClass();
    }

    // Note that we cannot accept Object[] because int[] (e.g.) does not
    // match.
    public NativeTensor(Object o) {
        if (!o.getClass().isArray()) {
            throw new IllegalArgumentException("Not an array");
        }
        Class<?> oClass = getAryType(o);
        if (oClass == String.class) {
            throw new NotImplementedException();
        } else if (Number.class.isAssignableFrom(oClass)) {
            genNumericTensor(o, (Class<Number>)oClass);
        } else {
            throw new IllegalArgumentException(
                    "Cannot convert type " + oClass.getSimpleName());
        }
    }
    
    private void genNumericTensor(Object ary, Class<Number> oClass) {
        this.shape = getShape(ary);
        int size = this.shape.stream().reduce(1, (a, b) -> a * b);
        NumType nt = typeMap.get(oClass);
        this.data = ByteBuffer.allocate(size * nt.size);
        this.data.order(ByteOrder.LITTLE_ENDIAN);
        this.type = nt.type;
        fillData(ary, 0, nt);
    }
    
    private void fillData(Object ary, int dim, NumType nt) {
        if (this.shape.get(dim) != Array.getLength(ary)) {
            throw new IllegalArgumentException("Array is not rectangular");
        }

        if (dim == this.shape.size() - 1) {
            // The innermost dimension can be copied directly.
            nt.putAndAdvance(data, ary);
            return;
        }

        for (int i = 0; i < this.shape.get(dim); i++) {
            fillData(Array.get(ary, i), dim + 1, nt);
        }
    }

    /*
     * Gets the shape of ary, which is assumed to be a rectangular array.
     */
    static List<Integer> getShape(Object ary) {
        int nDims = getDims(ary);
        List<Integer> shape = new ArrayList<Integer>(nDims);
        for (int i = 0; i < nDims; i++) {
            shape.add(Array.getLength(ary));
            ary = Array.get(ary, 0);
        }
        return shape;
    }
}
