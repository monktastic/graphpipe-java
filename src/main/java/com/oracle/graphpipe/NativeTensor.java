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

public abstract class NativeTensor {
    public static NativeTensor fromTensor(Tensor t) {
        if (t.type() == Type.String) {
            return new StringNativeTensor(t);
        } else {
            return new NumericNativeTensor(t);
        }
    }

    /**
     * @param ary An (arbitrary dimension) array of Numbers or Strings.
     * @throws ArrayIndexOutOfBoundsException If the final dimension contains
     * an empty array.
     */
    public static NativeTensor fromArray(Object ary) 
            throws ArrayIndexOutOfBoundsException {
        if (!ary.getClass().isArray()) {
            throw new IllegalArgumentException("Not an array");
        }
        Class<?> oClass = getAryType(ary);
        if (oClass == String.class) {
            return new StringNativeTensor(ary);
        } else if (Number.class.isAssignableFrom(oClass)) {
            return new NumericNativeTensor(ary, (Class<Number>)oClass);
        } else {
            throw new IllegalArgumentException(
                    "Cannot convert type " + oClass.getSimpleName());
        }
    }

    public static NativeTensor fromFlatArray(byte[] ary, long[] shape) {
        return new NumericNativeTensor(ary, shape, Byte.class);
    }

    public static NativeTensor fromFlatArray(short[] ary, long[] shape) {
        return new NumericNativeTensor(ary, shape, Short.class);
    }

    public static NativeTensor fromFlatArray(int[] ary, long[] shape) {
        return new NumericNativeTensor(ary, shape, Integer.class);
    }

    public static NativeTensor fromFlatArray(float[] ary, long[] shape) {
        return new NumericNativeTensor(ary, shape, Float.class);
    }

    public static NativeTensor fromFlatArray(double[] ary, long[] shape) {
        return new NumericNativeTensor(ary, shape, Double.class);
    }
    
    public static NativeTensor fromINDArray(INDArray ndAry) {
        return new NumericNativeTensor(ndAry);
    }

    public Tensor toTensor() {
        FlatBufferBuilder b = new FlatBufferBuilder(1024);
        int offset = this.Build(b);
        b.finish(offset);
        ByteBuffer bb = b.dataBuffer();
        return Tensor.getRootAsTensor(bb);
    }
    
    public abstract Object toFlatArray();
    public abstract INDArray toINDArray();
    public abstract int Build(FlatBufferBuilder b);
    
    // To primitive n-dim array.
    public abstract Object toNDArray();
    
    final List<Long> shape = new ArrayList<>();
    int elemCount = 1;

    private static Class<?> getAryType(Object ary) {
        if (ary.getClass().isArray()) {
            return getAryType(Array.get(ary, 0));
        } else {
            return ary.getClass();
        }
    }

    protected void fillShape(Tensor t) {
        for (int i = 0; i < t.shapeLength(); i++) {
            this.shape.add(t.shape(i));
            this.elemCount *= t.shape(i);
        }
    }

    protected void fillShape(Object ary) {
        if (ary.getClass().isArray()) {
            long length = Array.getLength(ary);
            this.shape.add(length);
            this.elemCount *= length;
            fillShape(Array.get(ary, 0));
        }
    }

    long[] shapeAsArray() {
        return this.shape.stream().mapToLong(i->i).toArray();
    }

    // NOTE: This is dangerous. If any of the dimensions is too big to fit in
    // an int, it will be truncated.
    int[] shapeAsIntArray() {
        return this.shape.stream().mapToInt(Long::intValue).toArray();
    }
}

class NumericNativeTensor extends NativeTensor {
    final ByteBuffer data;
    final NumConverter numConv;
    
    NumericNativeTensor(Tensor t) {
        fillShape(t);
        this.data = t.dataAsByteBuffer();
        this.numConv = NumConverters.byType(t.type());
    }
    
    NumericNativeTensor(Object ary, Class<? extends Number> clazz) {
        fillShape(ary);
        this.numConv = NumConverters.byClass((Class<Number>)clazz);
        this.data = ByteBuffer.allocate(this.elemCount * this.numConv.size)
                .order(ByteOrder.LITTLE_ENDIAN);
        fillFrom(ary, 0);
        this.data.rewind();
    }

    private void fillFrom(Object ary, int dim) {
        if (this.shape.get(dim) != Array.getLength(ary)) {
            throw new IllegalArgumentException("Array is not rectangular");
        }

        if (dim == this.shape.size() - 1) {
            this.numConv.put(data, ary);
            this.numConv.advance(data, ary);
        } else {
            for (int i = 0; i < this.shape.get(dim); i++) {
                fillFrom(Array.get(ary, i), dim + 1);
            }
        }
    }
    
    private void fillTo(Object ndAry, int dim) {
        if (dim == shape.size() - 1) {
            this.numConv.get(this.data, ndAry);
            this.numConv.advance(this.data, ndAry);
        } else {
            for (int i = 0; i < Array.getLength(ndAry); i++) {
                fillTo(Array.get(ndAry, i), dim + 1);
            }
        }
    }

    /**
     * From flat array.
     */
    NumericNativeTensor(
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
    
    NumericNativeTensor(INDArray ndAry) {
        this.data = ndAry.data().asNio();
        int size = ndAry.data().getElementSize();
        this.numConv = NumConverters.bySize(size);
        for (long s : ndAry.shape()) {
            this.shape.add(s);
        }
    }
    
    public INDArray toINDArray() {
        return this.numConv.buildINDArray(this);
    }

    public Object toFlatArray() {
        return this.numConv.toFlatArray(this);
    }

    public Object toNDArray() {
        Object ary = this.numConv.createNDArray(shapeAsIntArray());
        fillTo(ary, 0);
        return ary;
    }
    
    public int Build(FlatBufferBuilder b) {
        int shapeOffset = Tensor.createShapeVector(b, shapeAsArray());
        int dataOffset = Tensor.createDataVector(b, this.data.array());

        Tensor.startTensor(b);
        Tensor.addShape(b, shapeOffset);
        Tensor.addType(b, this.numConv.type);
        Tensor.addData(b, dataOffset);
        return Tensor.endTensor(b);
    }
}


class StringNativeTensor extends NativeTensor {
    final String[] data;
    
    StringNativeTensor(Tensor t) {
        fillShape(t);
        this.data = new String[t.stringValLength()];
        for (int i = 0; i < t.stringValLength(); i++) {
            this.data[i] = t.stringVal(i);
        }
    }
    
    StringNativeTensor(Object ary) {
        fillShape(ary);
        this.data = new String[this.elemCount];
        fillFrom(ary, 0, new int[]{0});
    }

    // We use int[] as an "int holder," because there's no standard mutable int.
    private void fillFrom(Object ary, int dim, int[] idx) {
        int len = Array.getLength(ary);
        if (this.shape.get(dim) != len) {
            throw new IllegalArgumentException("Array is not rectangular");
        }
        
        if (dim == this.shape.size() - 1) {
            System.arraycopy(ary, 0, this.data, idx[0], len);
            idx[0] += len;
        } else {
            for (int i = 0; i < this.shape.get(dim); i++) {
                fillFrom(Array.get(ary, i), dim + 1, idx);
            }
        }
    }
    
    private void fillTo(Object ary, int dim, int[] idx) {
        if (dim == this.shape.size() - 1) {
            int len = Array.getLength(ary);
            System.arraycopy(this.data, idx[0], ary, 0, len);
            idx[0] += len;
        } else {
            for (int i = 0; i < this.shape.get(dim); i++) {
                fillTo(Array.get(ary, i), dim + 1, idx);
            }
        }
    }
    
    public Object toNDArray() {
        Object ary = Array.newInstance(String.class, shapeAsIntArray());
        fillTo(ary, 0, new int[]{0});
        return ary;
    }
    
    public INDArray toINDArray() {
        throw new UnsupportedOperationException();
    }

    public Object toFlatArray() {
        return Arrays.copyOf(this.data, this.data.length, String[].class);
    }
    
    public int Build(FlatBufferBuilder b) {
        int shapeOffset = Tensor.createShapeVector(b, shapeAsArray());

        int[] stringOffsets = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            stringOffsets[i] = b.createString(data[i]);
        }
        int stringOffset = Tensor.createStringValVector(b, stringOffsets);

        Tensor.startTensor(b);
        Tensor.addShape(b, shapeOffset);
        Tensor.addType(b, Type.String);
        Tensor.addStringVal(b, stringOffset);
        return Tensor.endTensor(b);
    }
}