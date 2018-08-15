package com.oracle.graphpipe;

import com.oracle.graphpipefb.Type;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Created by aditpras on 8/7/18.
 */
abstract class NumConverter {
    final Class<? extends Number> clazz;
    final Class<?> primClazz;
    final int type;
    final int size;

    NumConverter(Class<? extends Number> clazz, Class<?> primClazz, int type, 
                 int size) {
        this.clazz = clazz;
        this.primClazz = primClazz;
        this.type = type;
        this.size = size;
    }

    abstract void get(ByteBuffer bb, Object ary);
    abstract void put(ByteBuffer bb, Object ary);

    void advance(ByteBuffer bb, Object ary) {
        bb.position(bb.position() + Array.getLength(ary) * this.size);
    }

    INDArray buildINDArray(NumericNativeTensor t) {
        // Only float[] and double[] are supported by Nd4j.
        throw new UnsupportedOperationException();
    }
    
    Object toFlatArray(NumericNativeTensor t) {
        Object ary = Array.newInstance(primClazz, t.elemCount);
        get(t.data, ary);
        return ary;
    }
    
    Object createNDArray(int[] shape) {
        return Array.newInstance(primClazz, shape);
    }

    static int[] shapeToIntAry(List<Long> shape) {
        return shape.stream().mapToInt(Long::intValue).toArray();
    }
}

class NumConverters {
    private static final List<NumConverter> all = new ArrayList<>();
    private static final Map<Class<? extends Number>, NumConverter> 
            byClass = new HashMap<>();
    private static final Map<Integer, NumConverter> byType = new HashMap<>();
    private static final Map<Integer, NumConverter> bySize = new HashMap<>();

    /*
    From the FlatBuffer def: 
    
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
    static {
        all.add(new NumConverter(Byte.class, byte.class, Type.Int8, 1) {
            void get(ByteBuffer bb, Object ary) {
                bb.get((byte[])ary);
            }
            void put(ByteBuffer bb, Object ary) {
                bb.put((byte[]) ary);
            }
            void advance(ByteBuffer bb, Object ary) {
                // No-op because put() already advances.
            }
        });
        all.add(new NumConverter(Short.class, short.class, Type.Int16, 2) {
            void get(ByteBuffer bb, Object ary) {
                bb.asShortBuffer().get((short[])ary);
            }
            void put(ByteBuffer bb, Object ary) {
                bb.asShortBuffer().put((short[]) ary);
            }
        });
        all.add(new NumConverter(Integer.class, int.class, Type.Int32, 4) {
            void get(ByteBuffer bb, Object ary) {
                bb.asIntBuffer().get((int[]) ary);
            }
            void put(ByteBuffer bb, Object ary) {
                bb.asIntBuffer().put((int[]) ary);
            }
        });
        all.add(new NumConverter(Long.class, long.class, Type.Int64, 8) {
            void get(ByteBuffer bb, Object ary) {
                bb.asLongBuffer().get((long[]) ary);
            }
            void put(ByteBuffer bb, Object ary) {
                bb.asLongBuffer().put((long[]) ary);
            }
        });
        all.add(new NumConverter(Float.class, float.class, Type.Float32, 4) {
            void get(ByteBuffer bb, Object ary) {
                bb.asFloatBuffer().get((float[]) ary);
            }
            void put(ByteBuffer bb, Object ary) {
                bb.asFloatBuffer().put((float[]) ary);
            }
            INDArray buildINDArray(NumericNativeTensor t) {
                return Nd4j.create(
                        (float[])toFlatArray(t), shapeToIntAry(t.shape));
            }
        });
        all.add(new NumConverter(Double.class, double.class, Type.Float64, 8) {
            void get(ByteBuffer bb, Object ary) {
                bb.asDoubleBuffer().get((double[]) ary);
            }
            void put(ByteBuffer bb, Object ary) {
                bb.asDoubleBuffer().put((double[]) ary);
            }
            INDArray buildINDArray(NumericNativeTensor t) {
                return Nd4j.create(
                        (double[])toFlatArray(t), shapeToIntAry(t.shape));
            }
        });
    }

    static {
        for (NumConverter nc : all) {
            byClass.put(nc.clazz, nc);
            byType.put(nc.type, nc);
            bySize.put(nc.size, nc);
        }
    }
    
    static NumConverter byClass(Class<? extends Number> clazz) {
        return byClass.get(clazz);
    }

    static NumConverter byType(int type) {
        return byType.get(type);
    }

    static NumConverter bySize(int size) {
        return bySize.get(size);
    }
}
