def if_dml(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with DirectML.

    Args:
      if_true: expression to evaluate if building with DirectML.
      if_false: expression to evaluate if building without DirectML.

    Returns:
      a select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        str(Label("//third_party/dml:using_dml")): if_true,
        "//conditions:default": if_false,
    })