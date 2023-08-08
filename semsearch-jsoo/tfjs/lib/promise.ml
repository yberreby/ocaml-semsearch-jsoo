(* Copied and adapted from Acid by @mefyl *)

open Js_of_ocaml

type 'a promise
type 't t = 't promise Js.t

let _Promise = Js.Unsafe.global##._Promise

let catch (p : 'a promise Js.t) ~(f : _ -> 'b) : 'b promise Js.t =
  (Js.Unsafe.coerce p)##_catch (Js.wrap_callback f)

let map (p : 'a promise Js.t) ~(f : 'a -> 'b) : 'b promise Js.t =
  (Js.Unsafe.coerce p)##_then (Js.wrap_callback f)

let return (v : 'a) : 'a promise Js.t =
  Js.Unsafe.meth_call _Promise "resolve" [| Js.Unsafe.inject v |]

let choose (promise_list : 'a promise Js.t Js.js_array Js.t) : 'a promise Js.t =
  Js.Unsafe.meth_call _Promise "race" [| Js.Unsafe.inject promise_list |]

let all (promise_list : 'a promise Js.t Js.js_array Js.t) : 'a promise Js.t =
  Js.Unsafe.meth_call _Promise "all" [| Js.Unsafe.inject promise_list |]

let to_lwt p =
  let res, resolve = Lwt.wait () in
  let _ = map p ~f:(fun v -> return @@ Lwt.wakeup resolve v) in
  res

let of_lwt p =
  new%js _Promise
    (Js.wrap_callback (fun resolve reject ->
         let ( >>= ) = Lwt.( >>= ) in
         let p () =
           p () >>= function
           | v ->
               Lwt.return @@ Js.Unsafe.fun_call resolve [| Js.Unsafe.inject v |]
         in
         Lwt.catch p (fun e ->
             let msg = Printexc.to_string e in
             let () =
               Js.Unsafe.fun_call reject
                 [| Js.Unsafe.inject (new%js Js.error_constr (Js.string msg)) |]
             in
             Lwt.return ())))
