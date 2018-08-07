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
    Class<? extends Number> clazz;
    int type;
    int size;

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

    INDArray buildINDArray(NativeTensor t) {
        throw new UnsupportedOperationException();
    }

    abstract Object toFlatArray(NativeTensor t);

    static int[] shapeToIntAry(List<Long> shape) {
        return shape.stream().mapToInt(Long::intValue).toArray();
    }
}

class NumConverters {
    static List<NumConverter> all = new ArrayList<>();
    private static Map<Class<? extends Number>, NumConverter> byClass = new HashMap<>();
    private static Map<Integer, NumConverter> byType = new HashMap<>();
    private static Map<Integer, NumConverter> bySize = new HashMap<>();

    static {
        all.add(new NumConverter(Byte.class, Type.Int8, 1) {
            void put(ByteBuffer bb, Object ary) {
                bb.put((byte[]) ary);
            }

            void advance(ByteBuffer bb, Object ary) {
                // No-op because put() already advances.
            }

            byte[] toFlatArray(NativeTensor t) {
                byte[] ary = new byte[t.elemCount];
                t.data.put(ary);
                return ary;
            }
        });
        all.add(new NumConverter(Short.class, Type.Int16, 2) {
            void put(ByteBuffer bb, Object ary) {
                bb.asShortBuffer().put((short[]) ary);
            }
            short[] toFlatArray(NativeTensor t) {
                short[] ary = new short[t.elemCount];
                t.data.asShortBuffer().put(ary);
                return ary;
            }
        });
        all.add(new NumConverter(Integer.class, Type.Int32, 4) {
            void put(ByteBuffer bb, Object ary) {
                bb.asIntBuffer().put((int[]) ary);
            }
            int[] toFlatArray(NativeTensor t) {
                int[] ary = new int[t.elemCount];
                t.data.asIntBuffer().put(ary);
                return ary;
            }
        });
        all.add(new NumConverter(Long.class, Type.Int64, 8) {
            void put(ByteBuffer bb, Object ary) {
                bb.asLongBuffer().put((long[]) ary);
            }
            long[] toFlatArray(NativeTensor t) {
                long[] ary = new long[t.elemCount];
                t.data.asLongBuffer().put(ary);
                return ary;
            }
        });
        all.add(new NumConverter(Float.class, Type.Float32, 4) {
            void put(ByteBuffer bb, Object ary) {
                bb.asFloatBuffer().put((float[]) ary);
            }
            float[] toFlatArray(NativeTensor t) {
                float[] ary = new float[t.elemCount];
                t.data.asFloatBuffer().put(ary);
                return ary;
            }

            INDArray buildINDArray(NativeTensor t) {
                float[] floats = new float[t.elemCount];
                t.data.asFloatBuffer().get(floats);
                return Nd4j.create(floats, shapeToIntAry(t.shape));
            }
        });
        all.add(new NumConverter(Double.class, Type.Float64, 8) {
            void put(ByteBuffer bb, Object ary) {
                bb.asDoubleBuffer().put((double[]) ary);
            }
            double[] toFlatArray(NativeTensor t) {
                double[] ary = new double[t.elemCount];
                t.data.asDoubleBuffer().get(ary);
                return ary;
            }

            INDArray buildINDArray(NativeTensor t) {
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
