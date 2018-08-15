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
    final int type;
    final int size;

    NumConverter(Class<? extends Number> clazz, int type, int size) {
        this.clazz = clazz;
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

    INDArray buildINDArray(NumericNativeTensor t) {
        throw new UnsupportedOperationException();
    }

    abstract Object toFlatArray(NumericNativeTensor t);

    static int[] shapeToIntAry(List<Long> shape) {
        return shape.stream().mapToInt(Long::intValue).toArray();
    }
}

class NumConverters {
    static final List<NumConverter> all = new ArrayList<>();
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
        all.add(new NumConverter(Byte.class, Type.Int8, 1) {
            void put(ByteBuffer bb, Object ary) {
                bb.put((byte[]) ary);
            }

            void advance(ByteBuffer bb, Object ary) {
                // No-op because put() already advances.
            }

            byte[] toFlatArray(NumericNativeTensor t) {
                byte[] ary = new byte[t.elemCount];
                t.data.get(ary);
                return ary;
            }
        });
        all.add(new NumConverter(Short.class, Type.Int16, 2) {
            void put(ByteBuffer bb, Object ary) {
                bb.asShortBuffer().put((short[]) ary);
            }
            short[] toFlatArray(NumericNativeTensor t) {
                short[] ary = new short[t.elemCount];
                t.data.asShortBuffer().get(ary);
                return ary;
            }
        });
        all.add(new NumConverter(Integer.class, Type.Int32, 4) {
            void put(ByteBuffer bb, Object ary) {
                bb.asIntBuffer().put((int[]) ary);
            }
            int[] toFlatArray(NumericNativeTensor t) {
                int[] ary = new int[t.elemCount];
                t.data.asIntBuffer().get(ary);
                return ary;
            }
        });
        all.add(new NumConverter(Long.class, Type.Int64, 8) {
            void put(ByteBuffer bb, Object ary) {
                bb.asLongBuffer().put((long[]) ary);
            }
            long[] toFlatArray(NumericNativeTensor t) {
                long[] ary = new long[t.elemCount];
                t.data.asLongBuffer().get(ary);
                return ary;
            }
        });
        all.add(new NumConverter(Float.class, Type.Float32, 4) {
            void put(ByteBuffer bb, Object ary) {
                bb.asFloatBuffer().put((float[]) ary);
            }
            float[] toFlatArray(NumericNativeTensor t) {
                float[] ary = new float[t.elemCount];
                t.data.asFloatBuffer().get(ary);
                return ary;
            }

            INDArray buildINDArray(NumericNativeTensor t) {
                float[] floats = new float[t.elemCount];
                t.data.asFloatBuffer().get(floats);
                return Nd4j.create(floats, shapeToIntAry(t.shape));
            }
        });
        all.add(new NumConverter(Double.class, Type.Float64, 8) {
            void put(ByteBuffer bb, Object ary) {
                bb.asDoubleBuffer().put((double[]) ary);
            }
            double[] toFlatArray(NumericNativeTensor t) {
                double[] ary = new double[t.elemCount];
                t.data.asDoubleBuffer().get(ary);
                return ary;
            }

            INDArray buildINDArray(NumericNativeTensor t) {
                double[] doubles = new double[t.elemCount];
                t.data.asDoubleBuffer().get(doubles);
                return Nd4j.create(doubles, shapeToIntAry(t.shape));
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
