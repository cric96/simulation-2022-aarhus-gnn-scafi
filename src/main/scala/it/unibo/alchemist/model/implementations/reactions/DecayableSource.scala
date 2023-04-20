package it.unibo.alchemist.model.implementations.reactions

import it.unibo.learning.abstractions.DecayReference

/** A trait that allows to attach decayable to a reaction. This will be update using internal decay logic (e.g., each
  * episode, at the end of the simulation, etc.)
  */
trait DecayableSource {

  protected var decayable = List.empty[(String, DecayReference[Any])]

  def attachDecayable(decay: (String, DecayReference[Any])*): Unit = decayable = (decay.toList) ::: decayable
}
