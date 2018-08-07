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
        fillData(ary, 0);
        this.data.rewind();
    }

    private void fillData(Object ary, int dim) {
        if (this.shape.get(dim) != Array.getLength(ary)) {
            throw new IllegalArgumentException("Array is not rectangular");
        }

        if (dim == this.shape.size() - 1) {
            // The innermost dimension can be copied directly.
            this.numConv.putAndAdvance(data, ary);
            return;
        }

        for (int i = 0; i < this.shape.get(dim); i++) {
            fillData(Array.get(ary, i), dim + 1);
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

    public int Build(FlatBufferBuilder b) {
        long[] shapeArray = this.shape.stream().mapToLong(i->i).toArray();
        int shapeOffset = Tensor.createShapeVector(b, shapeArray);
        int dataOffset = Tensor.createDataVector(b, this.data.array());

        Tensor.startTensor(b);
        Tensor.addShape(b, shapeOffset);
        Tensor.addType(b, this.numConv.type);
        Tensor.addData(b, dataOffset);
        return Tensor.endTensor(b);
    }
}


class StringNativeTensor extends NativeTensor {
    final List<String> data;
    
    StringNativeTensor(Tensor t) {
        fillShape(t);
        this.data = new ArrayList<>();
        for (int i = 0; i < t.stringValLength(); i++) {
            this.data.add(t.stringVal(i));
        }
    }
    
    StringNativeTensor(Object ary) {
        fillShape(ary);
        this.data = new ArrayList<>(this.elemCount);
        fillData(ary, 0);
    }

    private void fillData(Object ary, int dim) {
        if (this.shape.get(dim) != Array.getLength(ary)) {
            throw new IllegalArgumentException("Array is not rectangular");
        }

        for (int i = 0; i < this.shape.get(dim); i++) {
            Object child = Array.get(ary, i);
            if (child.getClass().isArray()) {
                fillData(child, dim + 1);
            } else {
                this.data.add((String)child);
            }
        }
    }

    public INDArray toINDArray() {
        throw new UnsupportedOperationException();
    }

    public Object toFlatArray() {
        Object[] ary = this.data.toArray();
        return Arrays.copyOf(ary, ary.length, String[].class);
    }
    
    public int Build(FlatBufferBuilder b) {
        long[] shapeArray = this.shape.stream().mapToLong(i->i).toArray();
        int shapeOffset = Tensor.createShapeVector(b, shapeArray);

        int[] stringOffsets = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            stringOffsets[i] = b.createString(data.get(i));
        }
        int stringOffset = Tensor.createStringValVector(b, stringOffsets);

        Tensor.startTensor(b);
        Tensor.addShape(b, shapeOffset);
        Tensor.addType(b, Type.String);
        Tensor.addStringVal(b, stringOffset);
        return Tensor.endTensor(b);
    }
}