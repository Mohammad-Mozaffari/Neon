def append_strings(head_string, variable_list, end_string_list):
    result = []
    if len(variable_list) == 0:
        result.append(head_string)
    else:
        if variable_list[0] == NO_DUPLICATE:
            for variable in variable_list[1:]:
                for end_string in end_string_list:
                    if variable in end_string:
                        result.append(f"{head_string}{variable}{end_string}")
        else:
            for varialbe in variable_list:
                if len(end_string_list) == 0:
                    result.append(f"{head_string}{varialbe}")
                else:
                    for end_string in end_string_list:
                        result.append(f"{head_string}{varialbe}{end_string}")
    return result


def instantiate_function(keyword_list):
    keyword_list.reverse()
    string_list = []
    variable_list = []
    for keyword in keyword_list:
        if type(keyword) is list:
            variable_list = keyword
        else:
            string_list = append_strings(keyword, variable_list, string_list)
    return string_list


NO_DUPLICATE = "NO_DUPLICATE"
EXTERN = False


if __name__ == "__main__":
    print("Fused Version")
    templates = instantiate_function(["{}template auto mapContainerFused<Neon::domain::".format("extern " if EXTERN else ""),
                            [NO_DUPLICATE, "dGrid"],
                            "::Field<",
                            [NO_DUPLICATE, "double", "float"],
                             ", 0>, ",
                            [NO_DUPLICATE, "double", "float"],
                            ", ",
                            [1, 2, 4, 8, 16, 32, 64], #Fusion Factor
                            ", ",
                            [1], #Flop Count
                            ", ",
                            [1], #Memory Accesses
                            ">(Neon::domain::",
                           ["dGrid"],
                           "::Field<",
                            [NO_DUPLICATE, "double", "float"],
                            ", 0>& pixels, ",
                            ["double", "float"],
                            " other_vals[]) -> Neon::set::Container;"])
    for template in templates:
        print(template)

    print("\n")

    templates = instantiate_function(["{}template auto mapContainer<Neon::domain::".format("extern " if EXTERN else ""),
                                      [NO_DUPLICATE, "dGrid"],
                                      "::Field<",
                                      [NO_DUPLICATE, "double", "float"],
                                      ", 0>, ",
                                      [NO_DUPLICATE, "double", "float"],
                                      ", ",
                                      [1], #Flop Count
                                      ", ",
                                      [1], #Memory Accesses
                                      ">(Neon::domain::",
                                      ["dGrid"],
                                      "::Field<",
                                      [NO_DUPLICATE, "double", "float"],
                                      ", 0>& pixels, ",
                                      ["double", "float"],
                                      " other_vals[]) -> Neon::set::Container;"])
    for template in templates:
        print(template)
