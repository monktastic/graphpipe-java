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
    
    int type;
    List<Integer> shape;
    List<String>  stringData;
    ByteBuffer data;
    
    static class NumType {
        int type;
        int size;
        PutNumber pn;
        
        NumType(int type, int size, PutNumber pn) {
            this.type = type;
            this.size = size;
            this.pn = pn;
        }
    }

    @FunctionalInterface
    interface PutNumber<T extends Number> {
        ByteBuffer apply(ByteBuffer bb, T number);
    }
   
    static Map<Class<? extends Number>, NumType> typeMap;
    static {
        // TODO(aprasad): Replace with method reference.
        PutNumber<Byte> pb = (bb, b) -> bb.put(b);
        PutNumber<Short> ps = (bb, s) -> bb.putShort(s);
        PutNumber<Integer> pi = (bb, i) -> bb.putInt(i);
        PutNumber<Long> pl = (bb, l) -> bb.putLong(l);
        PutNumber<Float> pf = (bb, f) -> bb.putFloat(f);
        PutNumber<Double> pd = (bb, d) -> bb.putDouble(d);
        
        typeMap = new HashMap<>();
        typeMap.put(Byte.class, new NumType(2, 1, pb));
        typeMap.put(Short.class, new NumType(4, 2, ps));
        typeMap.put(Integer.class, new NumType(6, 4, pi));
        typeMap.put(Long.class, new NumType(8, 8, pl));
        typeMap.put(Float.class, new NumType(10, 8, pf));
        typeMap.put(Double.class, new NumType(11, 16, pd));
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
        fillData(ary, nt);
    }
    
    private void fillData(Object ary, NumType nt) {
        int nDims = getDims(ary);
        _fillData(ary, 0, nt.pn);
    }
    
    private void _fillData(Object ary, int dim, PutNumber pn) {
        if (this.shape.get(dim) != Array.getLength(ary)) {
            throw new IllegalArgumentException("Array is not rectangular");
        }

        if (dim == this.shape.size() - 1) {
            for (int i = 0; i < Array.getLength(ary); i++) {
                Number n = (Number)Array.get(ary, i);
                pn.apply(data, n);
            }
            return;
        }

        for (int i = 0; i < this.shape.get(dim); i++) {
            _fillData(Array.get(ary, i), dim + 1, pn);
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
