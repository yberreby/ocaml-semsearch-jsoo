module Dtype = struct
  type t = Float32 | Int32 | Bool | Complex64 | String | Other of string

  let to_string = function
    | Float32 -> "float32"
    | Int32 -> "int32"
    | Bool -> "bool"
    | Complex64 -> "complex64"
    | String -> "string"
    | Other s -> s
end
