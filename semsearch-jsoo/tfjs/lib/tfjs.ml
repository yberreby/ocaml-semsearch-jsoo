open Js_of_ocaml
include Tfjs_intf

type model
type io_handler

module Tensor = struct
  type t

  let to_array tensor : float Array.t =
    let output =
      Js.Unsafe.meth_call (Js.Unsafe.inject tensor) "dataSync" [||]
    in
    Js.to_array output
end

let tf = lazy (Js.Unsafe.js_expr "require('@tensorflow/tfjs')")
let tfn = lazy (Js.Unsafe.js_expr "require('@tensorflow/tfjs-node')")

let load_graph_model handler =
  Js.Unsafe.fun_call
    (Lazy.force tf)##.loadGraphModel
    [| Js.Unsafe.inject handler |]
  |> Promise.to_lwt

let tensor2d data shape dtype =
  Js.Unsafe.fun_call
    (Lazy.force tf)##.tensor2d
    [|
      Js.Unsafe.inject data;
      Js.array shape |> Js.Unsafe.inject;
      Dtype.to_string dtype |> Js.string |> Js.Unsafe.inject;
    |]

let zeros shape dtype =
  Js.Unsafe.fun_call
    (Lazy.force tf)##.zeros
    [|
      Js.array shape |> Js.Unsafe.inject;
      Dtype.to_string dtype |> Js.string |> Js.Unsafe.inject;
    |]

let matmul tensor1 tensor2 =
  Js.Unsafe.fun_call
    (Lazy.force tf)##.matMul
    [| Js.Unsafe.inject tensor1; Js.Unsafe.inject tensor2 |]

let transpose tensor =
  Js.Unsafe.fun_call (Lazy.force tf)##.transpose [| Js.Unsafe.inject tensor |]

let reshape tensor (shape : int array) =
  let shape_js = Js.array shape in
  Js.Unsafe.meth_call tensor "reshape" [| Js.Unsafe.inject shape_js |]

module IO = struct
  let fileSystem path =
    Js.Unsafe.fun_call
      (Lazy.force tfn)##.io##.fileSystem
      [| path |> Js.string |> Js.Unsafe.inject |]
end
