package com.oracle.graphpipe;

import com.google.flatbuffers.FlatBufferBuilder;
import com.oracle.graphpipefb.Tensor;
import com.oracle.graphpipefb.Type;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
    public NativeTensor(Tensor t) {
        for (int i = 0; i < t.shapeLength(); i++) {
            shape.add(t.shape(i));
            this.elemCount *= t.shape(i);
        }
        if (t.type() == Type.String) {
            for (int i = 0; i < t.stringValLength(); i++) {
                this.stringData.add(t.stringVal(i));
            }
        } else {
            this.data = t.dataAsByteBuffer();
            this.numConv = NumConverters.byType(t.type());
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
            this.numConv = NumConverters.byClass((Class<Number>)oClass);
            genNumericTensor(ary);
        } else {
            throw new IllegalArgumentException(
                    "Cannot convert type " + oClass.getSimpleName());
        }
    }

    public NativeTensor(byte[] ary, long[] shape) {
        initFromFlatArrayNumeric(ary, shape, Byte.class);
    }

    public NativeTensor(short[] ary, long[] shape) {
        initFromFlatArrayNumeric(ary, shape, Short.class);
    }

    public NativeTensor(int[] ary, long[] shape) {
        initFromFlatArrayNumeric(ary, shape, Integer.class);
    }

    public NativeTensor(float[] ary, long[] shape) {
        initFromFlatArrayNumeric(ary, shape, Float.class);
    }

    public NativeTensor(double[] ary, long[] shape) {
        initFromFlatArrayNumeric(ary, shape, Double.class);
    }
    
    public NativeTensor(INDArray ndAry) {
        this.data = ndAry.data().asNio();
        int size = ndAry.data().getElementSize();
        this.numConv = NumConverters.bySize(size);
        for (long s : ndAry.shape()) {
            this.shape.add(s);
        }
    }

    public INDArray toINDArray() {
        if (this.numConv != null) {
            return this.numConv.buildINDArray(this);
        } else {
            throw new UnsupportedOperationException();
        }
    }

    public Tensor toTensor() {
        FlatBufferBuilder b = new FlatBufferBuilder(1024);
        int offset = this.Build(b);
        b.finish(offset);
        ByteBuffer bb = b.dataBuffer();
        return Tensor.getRootAsTensor(bb);
    }
    
    public Object toFlatArray() {
        if (this.numConv != null) {
            return this.numConv.toFlatArray(this);
        } else {
            Object[] ary = this.stringData.toArray(); 
            return Arrays.copyOf(ary, ary.length, String[].class);
        }
    }

    private int BuildNumeric(FlatBufferBuilder b) {
        long[] shapeArray = this.shape.stream().mapToLong(i->i).toArray();
        int shapeOffset = Tensor.createShapeVector(b, shapeArray);
        int dataOffset = Tensor.createDataVector(b, this.data.array());

        Tensor.startTensor(b);
        Tensor.addShape(b, shapeOffset);
        Tensor.addType(b, this.numConv.type);
        Tensor.addData(b, dataOffset);
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
        Tensor.addShape(b, shapeOffset);
        Tensor.addType(b, Type.String);
        Tensor.addStringVal(b, stringOffset);
        return Tensor.endTensor(b);
    }
    
    public int Build(FlatBufferBuilder b) {
        if (numConv != null) {
            return BuildNumeric(b);
        } else {
            return BuildString(b);
        }
    }
    
    List<Long> shape = new ArrayList<>();
    int elemCount = 1;
    List<String> stringData;
    ByteBuffer data;
    NumConverter numConv;

    private static Class<?> getAryType(Object ary) {
        if (ary.getClass().isArray()) {
            return getAryType(Array.get(ary, 0));
        } else {
            return ary.getClass();
        }
    }

    private void initFromFlatArrayNumeric(
            Object ary, long[] shape, Class<? extends Number> clazz) {
        for (long s : shape) {
            this.shape.add(s);
            this.elemCount *= s;
        }
        this.numConv = NumConverters.byClass(clazz);
        this.data = ByteBuffer
                .allocate(this.elemCount * this.numConv.size)
                .order(ByteOrder.LITTLE_ENDIAN);
        this.numConv.put(this.data, ary);
        this.data.rewind();
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
        this.data = ByteBuffer.allocate(this.elemCount * this.numConv.size)
                .order(ByteOrder.LITTLE_ENDIAN);
        fillNumericData(ary, 0);
        this.data.rewind();
    }
    
    private void fillNumericData(Object ary, int dim) {
        if (this.shape.get(dim) != Array.getLength(ary)) {
            throw new IllegalArgumentException("Array is not rectangular");
        }

        if (dim == this.shape.size() - 1) {
            // The innermost dimension can be copied directly.
            this.numConv.putAndAdvance(data, ary);
            return;
        }

        for (int i = 0; i < this.shape.get(dim); i++) {
            fillNumericData(Array.get(ary, i), dim + 1);
        }
    }

    private void fillShape(Object ary) {
        if (ary.getClass().isArray()) {
            long length = Array.getLength(ary);
            this.shape.add(length);
            this.elemCount *= length;
            fillShape(Array.get(ary, 0));
        }
    }
}
