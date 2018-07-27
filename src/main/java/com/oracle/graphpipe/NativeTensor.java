package com.oracle.graphpipe;

import com.google.flatbuffers.FlatBufferBuilder;
import com.oracle.graphpipefb.Tensor;
import com.oracle.graphpipefb.Type;

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
    
    private List<Integer> shape = new ArrayList<>();
    private int elemCount = 1;
    List<String> stringData;
    ByteBuffer data;
    private NumType numType;
    
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
    
    private static NumType byteNt = new NumType(Type.Int8, 1) {
        void put(ByteBuffer bb, Object ary) {
           bb.put((byte[])ary); 
        }
        void advance(ByteBuffer bb, Object ary) {
            // No-op because put() already advances.
        }
    };
    private static NumType shortNt = new NumType(Type.Int16, 2) {
        void put(ByteBuffer bb, Object ary) {
            bb.asShortBuffer().put((short[])ary);
        }
    };
    private static NumType intNt = new NumType(Type.Int32, 4) {
        void put(ByteBuffer bb, Object ary) {
            bb.asIntBuffer().put((int[])ary);
        }
    };
    private static NumType longNt = new NumType(Type.Int32, 8) {
        void put(ByteBuffer bb, Object ary) {
            bb.asLongBuffer().put((long[])ary);
        }
    };
    private static NumType floatNt = new NumType(Type.Float32, 8) {
        void put(ByteBuffer bb, Object ary) {
            bb.asFloatBuffer().put((float[])ary);
        }
    };
    private static NumType doubleNt = new NumType(Type.Float64, 16) {
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
    
    public ByteBuffer makeTensorByteBuffer() {
        FlatBufferBuilder b = new FlatBufferBuilder(1024);
        int offset = this.Build(b);
        b.finish(offset);
        return b.dataBuffer();
    }
    
    private int BuildNumeric(FlatBufferBuilder b) {
        long[] shapeArray = this.shape.stream().mapToLong(i->i).toArray();
        int shapeOffset = Tensor.createShapeVector(b, shapeArray);
        int dataOffset = Tensor.createDataVector(b, this.data.array());

        Tensor.startTensor(b);
        Tensor.addData(b, dataOffset);
        Tensor.addShape(b, shapeOffset);
        Tensor.addType(b, this.numType.type);
        return Tensor.endTensor(b);
    }
    
    private int BuildString(FlatBufferBuilder b) {
        long[] shapeArray = this.shape.stream().mapToLong(i->i).toArray();
        int shapeOffset = Tensor.createShapeVector(b, shapeArray);
      
        int[] stringOffsets = new int[stringData.size()];
        for (int i = 0; i < stringData.size(); i++) {
            stringOffsets[i] = b.createString(stringData.get(i));
        }
        int stringOffset = Tensor.createStringValVector(b, stringOffsets);
        
        Tensor.startTensor(b);
        Tensor.addStringVal(b, stringOffset);
        Tensor.addShape(b, shapeOffset);
        Tensor.addType(b, Type.String);
        return Tensor.endTensor(b);
    }
    
    public int Build(FlatBufferBuilder b) {
        if (numType != null) {
            return BuildNumeric(b);
        } else {
            return BuildString(b);
        }
    }
   
    private static Class<?> getAryType(Object ary) {
        if (ary.getClass().isArray()) {
            return getAryType(Array.get(ary, 0));
        } else {
            return ary.getClass();
        }
    }

    /**
     * @param ary An (arbitrary dimension) array of Numbers or Strings.
     * @throws ArrayIndexOutOfBoundsException If the final dimension contains
     * an empty array.
     */
    public NativeTensor(Object ary) throws ArrayIndexOutOfBoundsException {
        if (!ary.getClass().isArray()) {
            throw new IllegalArgumentException("Not an array");
        }
        Class<?> oClass = getAryType(ary);
        fillShape(ary);
        if (oClass == String.class) {
            genStringTensor(ary);
        } else if (Number.class.isAssignableFrom(oClass)) {
            this.numType = typeMap.get(oClass);
            genNumericTensor(ary);
        } else {
            throw new IllegalArgumentException(
                    "Cannot convert type " + oClass.getSimpleName());
        }
    }
    
    private void genStringTensor(Object ary) {
        this.stringData = new ArrayList<>(this.elemCount);
        fillStringData(ary, 0);
    }

    private void fillStringData(Object ary, int dim) {
        if (this.shape.get(dim) != Array.getLength(ary)) {
            throw new IllegalArgumentException("Array is not rectangular");
        }

        for (int i = 0; i < this.shape.get(dim); i++) {
            Object child = Array.get(ary, i);
            if (child.getClass().isArray()) {
                fillStringData(child, dim + 1);
            } else {
                this.stringData.add((String)child);
            }
        }
    }

    private void genNumericTensor(Object ary) {
        this.data = ByteBuffer.allocate(this.elemCount * this.numType.size)
                .order(ByteOrder.LITTLE_ENDIAN);
        fillNumericData(ary, 0);
    }
    
    private void fillNumericData(Object ary, int dim) {
        if (this.shape.get(dim) != Array.getLength(ary)) {
            throw new IllegalArgumentException("Array is not rectangular");
        }

        if (dim == this.shape.size() - 1) {
            // The innermost dimension can be copied directly.
            this.numType.putAndAdvance(data, ary);
            return;
        }

        for (int i = 0; i < this.shape.get(dim); i++) {
            fillNumericData(Array.get(ary, i), dim + 1);
        }
    }

    private void fillShape(Object ary) {
        if (ary.getClass().isArray()) {
            int length = Array.getLength(ary);
            this.shape.add(length);
            this.elemCount *= length;
            fillShape(Array.get(ary, 0));
        }
    }
}
