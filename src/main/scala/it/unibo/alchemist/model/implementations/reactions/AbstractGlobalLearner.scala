package it.unibo.alchemist.model.implementations.reactions

import it.unibo.learning.abstractions.DecayReference

trait AbstractGlobalLearner {
  protected var decayable = List.empty[(String, DecayReference[Any])]

  def attachDecayable(decay: (String, DecayReference[Any])*): Unit = decayable = (decay.toList) ::: decayable
}
