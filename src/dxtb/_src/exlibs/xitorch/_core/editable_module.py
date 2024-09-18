# This file is part of dxtb, modified from xitorch/xitorch.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Original file licensed under the MIT License by xitorch/xitorch.
# Modifications made by Grimme Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import inspect
import warnings
from abc import abstractmethod
from typing import Dict, List, Sequence, Union

import torch

from dxtb._src.exlibs.xitorch._utils.attr import del_attr, get_attr, set_attr
from dxtb._src.exlibs.xitorch._utils.exceptions import GetSetParamsError

__all__ = ["EditableModule"]

torch_float_type = [torch.float32, torch.float, torch.float64, torch.float16]


class EditableModule:
    """
    ``EditableModule`` is a base class to enable classes that it inherits be
    converted to pure functions for higher order derivatives purpose.
    """

    def getparams(self, methodname: str) -> Sequence[torch.Tensor]:
        # Returns a list of tensor parameters used in the object's operations

        paramnames = self.cached_getparamnames(methodname)
        return [get_attr(self, name) for name in paramnames]

    def setparams(self, methodname: str, *params) -> int:
        # Set the input parameters to the object's parameters to make a copy of
        # the operations.
        # *params is an excessive list of the parameters to be set and the
        # method will return the number of parameters it sets.
        paramnames = self.cached_getparamnames(methodname)
        for name, val in zip(paramnames, params):
            try:
                set_attr(self, name, val)
            except TypeError as e:  # failed because val should be param
                del_attr(self, name)
                set_attr(self, name, val)

        return len(params)

    def cached_getparamnames(
        self, methodname: str, refresh: bool = False
    ) -> List[str]:
        # getparamnames, but cached, so it is only called once
        if not hasattr(self, "_paramnames_"):
            self._paramnames_: Dict[str, List[str]] = {}

        if methodname not in self._paramnames_:
            self._paramnames_[methodname] = self.getparamnames(methodname)
        return self._paramnames_[methodname]

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        This method should list tensor names that affect the output of the
        method with name indicated in ``methodname``.
        If the ``methodname`` is not on the list in this function, it should
        raise ``KeyError``.

        Arguments
        ---------
        methodname: str
            The name of the method of the class.
        prefix: str
            The prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        Sequence of string
            Sequence of name of parameters affecting the output of the method.

        Raises
        ------
        KeyError
            If the list in this function does not contain ``methodname``.

        Example
        -------
        .. testsetup::

            import torch
            import xitorch

        .. doctest::

            >>> class A(dxtb._src.exlibs.xitorch.EditableModule):
            ...     def __init__(self, a):
            ...         self.b = a*a
            ...
            ...     def mult(self, x):
            ...         return self.b * x
            ...
            ...     def getparamnames(self, methodname, prefix=""):
            ...         if methodname == "mult":
            ...             return [prefix+"b"]
            ...         else:
            ...             raise KeyError()
        """
        pass

    def getuniqueparams(
        self, methodname: str, onlyleaves: bool = False
    ) -> List[torch.Tensor]:
        """
        Returns the list of unique parameters involved in the method specified
        by `methodname`.

        Arguments
        ---------
        methodname: str
            Name of the method where the returned parameters play roles.
        onlyleaves: bool
            If True, only returns leaf tensors. Otherwise, returns all tensors.

        Returns
        -------
        list of tensors
            List of tensors that are involved in the specified method of the
            object.
        """
        allparams = self.getparams(methodname)
        idxs = self._get_unique_params_idxs(methodname, allparams)
        if onlyleaves:
            return [allparams[i] for i in idxs if allparams[i].is_leaf]
        else:
            return [allparams[i] for i in idxs]

    def setuniqueparams(self, methodname: str, *uniqueparams) -> int:
        nparams = self._number_of_params[methodname]
        allparams = [None for _ in range(nparams)]
        maps = self._unique_params_maps[methodname]

        for j in range(len(uniqueparams)):
            jmap = maps[j]
            p = uniqueparams[j]
            for i in jmap:
                allparams[i] = p

        return self.setparams(methodname, *allparams)

    def _get_unique_params_idxs(
        self,
        methodname: str,
        allparams: Union[Sequence[torch.Tensor], None] = None,
    ) -> Sequence[int]:
        if not hasattr(self, "_unique_params_idxs"):
            self._unique_params_idxs = {}  # type: Dict[str,Sequence[int]]
            self._unique_params_maps = {}
            self._number_of_params = {}

        if methodname in self._unique_params_idxs:
            return self._unique_params_idxs[methodname]
        if allparams is None:
            allparams = self.getparams(methodname)

        # get the unique ids
        ids = []  # type: List[int]
        idxs = []
        idx_map = []  # type: List[List[int]]
        for i in range(len(allparams)):
            param = allparams[i]
            id_param = id(param)

            # search the id if it has been added to the list
            if id_param in ids:
                jfound = ids.index(id_param)
                idx_map[jfound].append(i)
            else:
                ids.append(id_param)
                idxs.append(i)
                idx_map.append([i])

        self._number_of_params[methodname] = len(allparams)
        self._unique_params_idxs[methodname] = idxs
        self._unique_params_maps[methodname] = idx_map
        return idxs

    ############# debugging #############
    def assertparams(self, method, *args, **kwargs):
        """
        Perform a rigorous check on the implemented ``getparamnames``
        in the class for a given method and its arguments as well as keyword
        arguments.
        It raises warnings if there are missing or excess parameters in the
        ``getparamnames`` implementation.

        Arguments
        ---------
        method: callable method
            The method of this class to be tested
        *args:
            Arguments of the method
        **kwargs:
            Keyword arguments of the method

        Example
        -------
        .. testsetup:: assertparams

            import torch
            import xitorch
            import sys
            sys.stderr = sys.stdout

        .. doctest:: assertparams

            >>> class AClass(dxtb._src.exlibs.xitorch.EditableModule):
            ...     def __init__(self, a):
            ...         self.a = a
            ...         self.b = a*a
            ...
            ...     def mult(self, x):
            ...         return self.b * x
            ...
            ...     def getparamnames(self, methodname, prefix=""):
            ...         if methodname == "mult":
            ...             return [prefix+"a"]  # intentionally wrong
            ...         else:
            ...             raise KeyError()
            >>> a = torch.tensor(2.0).requires_grad_()
            >>> x = torch.tensor(0.4).requires_grad_()
            >>> A = AClass(a)
            >>> A.assertparams(A.mult, x) # doctest:+ELLIPSIS
            <...>:1: UserWarning: getparams for AClass.mult does not include: b
              A.assertparams(A.mult, x) # doctest:+ELLIPSIS
            <...>:1: UserWarning: getparams for AClass.mult has excess parameters: a
              A.assertparams(A.mult, x) # doctest:+ELLIPSIS
            "mult" method check done
        """
        # check the method input
        if not inspect.ismethod(method):
            raise TypeError("The input method must be a method")
        methodself = method.__self__
        if methodself is not self:
            raise RuntimeError(
                "The method does not belong to the same instance"
            )

        methodname = method.__name__

        # assert if the method preserve the float tensors of the object
        self.__assert_method_preserve(method, *args, **kwargs)
        self.__assert_get_correct_params(
            method, *args, **kwargs
        )  # check if getparams returns the correct tensors
        print('"%s" method check done' % methodname)

    def __assert_method_preserve(self, method, *args, **kwargs):
        # this method assert if method does not change the float tensor parameters
        # of the object (i.e. it preserves the state of the object)

        all_params0, names0 = _get_tensors(self)
        all_params0 = [p.clone() for p in all_params0]
        method(*args, **kwargs)
        all_params1, names1 = _get_tensors(self)

        # now assert if all_params0 == all_params1
        clsname = method.__self__.__class__.__name__
        methodname = method.__name__
        msg = "The method {}.{} does not preserve the object's float tensors: \n".format(
            clsname,
            methodname,
        )
        if len(all_params0) != len(all_params1):
            msg += "The number of parameters changed:\n"
            msg += "* number of object's parameters before: %d\n" % len(
                all_params0
            )
            msg += "* number of object's parameters after : %d\n" % len(
                all_params1
            )
            raise GetSetParamsError(msg)

        for pname, p0, p1 in zip(names0, all_params0, all_params1):
            if p0.shape != p1.shape:
                msg += "The shape of %s changed\n" % pname
                msg += f"* (before) {pname}.shape: {p0.shape}\n"
                msg += f"* (after ) {pname}.shape: {p1.shape}\n"
                raise GetSetParamsError(msg)
            if not torch.allclose(p0, p1):
                msg += "The value of %s changed\n" % pname
                msg += f"* (before) {pname}: {p0}\n"
                msg += f"* (after ) {pname}: {p1}\n"
                raise GetSetParamsError(msg)

    def __assert_get_correct_params(self, method, *args, **kwargs):
        # this function perform checks if the getparams on the method returns
        # the correct tensors

        methodname = method.__name__
        clsname = method.__self__.__class__.__name__

        # get all tensor parameters in the object
        # all_params, all_names = _get_tensors(self)

        # def _get_tensor_name(param):
        #     for i in range(len(all_params)):
        #         if id(all_params[i]) == id(param):
        #             return all_names[i]
        #     return None

        # get the parameter tensors used in the operation and the tensors specified by the developer
        oper_names, oper_params = self.__list_operating_params(
            method, *args, **kwargs
        )
        user_names = self.getparamnames(method.__name__)
        user_params = [get_attr(self, name) for name in user_names]
        user_params_id = [id(p) for p in user_params]
        oper_params_id = [id(p) for p in oper_params]
        user_params_id_set = set(user_params_id)
        oper_params_id_set = set(oper_params_id)

        # check if the userparams contains non-tensor
        for i in range(len(user_params)):
            param = user_params[i]
            if (not isinstance(param, torch.Tensor)) or (
                isinstance(param, torch.Tensor)
                and param.dtype not in torch_float_type
            ):
                msg = (
                    "Parameter %s is a non-floating point tensor"
                    % user_names[i]
                )
                raise GetSetParamsError(msg)

        # check if there are missing parameters (present in operating params, but not in the user params)
        missing_names = []
        for i in range(len(oper_names)):
            if oper_params_id[i] not in user_params_id_set:
                # if oper_names[i] not in user_names:
                missing_names.append(oper_names[i])
        # if there are missing parameters, give a warning (because the program
        # can still run correctly, e.g. missing parameters are parameters that
        # are never set to require grad)
        if len(missing_names) > 0:
            msg = "getparams for {}.{} does not include: {}".format(
                clsname,
                methodname,
                ", ".join(missing_names),
            )
            warnings.warn(msg, stacklevel=3)

        # check if there are excessive parameters (present in the user params, but not in the operating params)
        excess_names = []
        for i in range(len(user_names)):
            if user_params_id[i] not in oper_params_id_set:
                # if user_names[i] not in oper_names:
                excess_names.append(user_names[i])
        # if there are excess parameters, give warnings
        if len(excess_names) > 0:
            msg = "getparams for {}.{} has excess parameters: {}".format(
                clsname,
                methodname,
                ", ".join(excess_names),
            )
            warnings.warn(msg, stacklevel=3)

    def __list_operating_params(self, method, *args, **kwargs):
        # Sequence the tensors used in executing the method by calling the method
        # and see which parameters are connected in the backward graph

        # get all the tensors recursively
        all_tensors, all_names = _get_tensors(self)

        # copy the tensors and require them to be differentiable
        copy_tensors0 = [
            tensor.clone().detach().requires_grad_() for tensor in all_tensors
        ]
        copy_tensors = copy.copy(copy_tensors0)
        _set_tensors(self, copy_tensors)

        # run the method and see which one has the gradients
        output = method(*args, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                "The method to be asserted must have a tensor output"
            )
        output = output.sum()
        grad_tensors = torch.autograd.grad(
            output, copy_tensors0, retain_graph=True, allow_unused=True
        )

        # return the original tensor
        all_tensors_copy = copy.copy(all_tensors)
        _set_tensors(self, all_tensors_copy)

        names = []
        params = []
        for i, grad in enumerate(grad_tensors):
            if grad is None:
                continue
            names.append(all_names[i])
            params.append(all_tensors[i])

        return names, params


############################ traversing functions ############################
def _traverse_obj(obj, prefix, action, crit, max_depth=20, exception_ids=None):
    """
    Traverse an object to get/set variables that are accessible through the object.
    """
    if exception_ids is None:
        # None is set as default arg to avoid expanding list for multiple
        # invokes of _get_tensors without exception_ids argument
        exception_ids = set()

    if isinstance(obj, torch.nn.Module):
        generators = [obj._parameters.items(), obj._modules.items()]
        name_format = "{prefix}{key}"
        objdicts = [obj._parameters, obj._modules]
    elif hasattr(obj, "__dict__"):
        generators = [obj.__dict__.items()]
        name_format = "{prefix}{key}"
        objdicts = [obj.__dict__]
    elif hasattr(obj, "__iter__"):
        generators = [obj.items() if isinstance(obj, dict) else enumerate(obj)]
        name_format = "{prefix}[{key}]"
        objdicts = [obj]
    else:
        raise RuntimeError("The object must be iterable or keyable")

    for generator, objdict in zip(generators, objdicts):
        for key, elmt in generator:
            name = name_format.format(prefix=prefix, key=key)
            if crit(elmt):
                action(elmt, name, objdict, key)
                continue

            hasdict = hasattr(elmt, "__dict__")
            hasiter = hasattr(elmt, "__iter__")
            if hasdict or hasiter:
                # add exception to avoid infinite loop if there is a mutual dependant on objects
                if id(elmt) in exception_ids:
                    continue
                else:
                    exception_ids.add(id(elmt))

                if max_depth > 0:
                    _traverse_obj(
                        elmt,
                        action=action,
                        crit=crit,
                        prefix=name + "." if hasdict else name,
                        max_depth=max_depth - 1,
                        exception_ids=exception_ids,
                    )
                else:
                    raise RecursionError("Maximum number of recursion reached")


def _get_tensors(obj, prefix="", max_depth=20):
    """
    Collect all tensors in an object recursively and return the tensors as well
    as their "names" (names meaning the address, e.g. "self.a[0].elmt").

    Arguments
    ---------
    * obj: an instance
        The object user wants to traverse down
    * prefix: str
        Prefix of the name of the collected tensors. Default: ""

    Returns
    -------
    * res: list of torch.Tensor
        Sequence of tensors collected recursively in the object.
    * name: list of str
        Sequence of names of the collected tensors.
    """

    # get the tensors recursively towards torch.nn.Module
    res = []
    names = []

    def action(elmt, name, objdict, key):
        res.append(elmt)
        names.append(name)

    # traverse down the object to collect the tensors
    crit = (
        lambda elmt: isinstance(elmt, torch.Tensor)
        and elmt.dtype in torch_float_type
    )
    _traverse_obj(
        obj, action=action, crit=crit, prefix=prefix, max_depth=max_depth
    )
    return res, names


def _set_tensors(obj, all_params, max_depth=20):
    """
    Set the tensors in an object to new tensor object listed in `all_params`.

    Arguments
    ---------
    * obj: an instance
        The object user wants to traverse down
    * all_params: list of torch.Tensor
        Sequence of tensors to be put in the object.
    * max_depth: int
        Maximum recursive depth to avoid infinitely running program.
        If the maximum depth is reached, then raise a RecursionError.
    """

    def action(elmt, name, objdict, key):
        objdict[key] = all_params.pop(0)

    # traverse down the object to collect the tensors
    crit = (
        lambda elmt: isinstance(elmt, torch.Tensor)
        and elmt.dtype in torch_float_type
    )
    _traverse_obj(obj, action=action, crit=crit, prefix="", max_depth=max_depth)
