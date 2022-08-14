package it.unibo.alchemist.boundary.swingui.effect.impl;

import it.unibo.alchemist.boundary.swingui.effect.api.LayerToFunctionMapper;
import it.unibo.alchemist.model.implementations.layers.DensityMap;
import it.unibo.alchemist.model.interfaces.Layer;
import it.unibo.alchemist.model.interfaces.Position2D;

import java.util.Collection;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DensityMapFunction implements LayerToFunctionMapper {
    @Override
    public <T, P extends Position2D<P>> Stream<Function<? super P, ? extends Number>> map(Stream<Layer<T, P>> stream) {

        return stream.filter(layer -> layer instanceof DensityMap)
                .map(layer -> (DensityMap<P>) layer)
                .map(layer -> layer::getValue);
    }

    @Override
    public <T, P extends Position2D<P>> Collection<Function<? super P, ? extends Number>> map(Collection<? extends Layer<T, P>> collection) {
        Stream<Layer<T, P>> layers = (Stream<Layer<T, P>>) collection.stream();
        return map(layers).collect(Collectors.toList());
    }
}
