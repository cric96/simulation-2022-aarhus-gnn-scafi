package it.unibo.scafi

class ExplorerCollective extends Agent {

  override def main(): Any = {
    val state = computeState
    node.put("state", state)
    // node.put("action", policy(state))
  }

}
