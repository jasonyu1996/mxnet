# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License


# pylint: disable=invalid-name, too-many-locals, too-many-branches, too-many-arguments


import re
from collections import Counter
from ..hybrid_op import _register_hybrid_op

_EINSUM_PATTERN = re.compile(r"^([a-z]+)((?:,[a-z]+)*)(?:->([a-z]*))?$")


def _reduce(F, dat, ins, keep):
    invert_map = dict(zip(ins, range(0, len(ins))))
    axes_to_reduce = [invert_map[sub] for sub in set(ins) - keep]
    if axes_to_reduce:
        dat = F.sum(dat, axis=axes_to_reduce)
        for i in sorted(axes_to_reduce, reverse=True):
            del ins[i]
    return dat, ins

def _contract_pair(F, a, b, a_sub, b_sub, keep):
    na = len(a_sub)
    nb = len(b_sub)
    ia, ib = (0, 0)
    new_ins = []
    shape_a = []
    shape_b = []
    while ia < na or ib < nb:
        if ia < na and (ib == nb or a_sub[ia] <= b_sub[ib]):
            if ib < nb and a_sub[ia] == b_sub[ib]:
                if a.shape[ia] != b.shape[ib]:
                    raise ValueError('Input axes with the same subscript should be equal!')
                size_a = size_b = a.shape[ia]
                new_ins.append(a_sub[ia])
                ia += 1
                ib += 1
            else:
                size_a = a.shape[ia]
                size_b = 1
                new_ins.append(a_sub[ia])
                ia += 1
        else:
            size_a = 1
            size_b = b.shape[ib]
            new_ins.append(b_sub[ib])
            ib += 1
        shape_a.append(size_a)
        shape_b.append(size_b)

    na = F.reshape(a, shape=shape_a)
    nb = F.reshape(b, shape=shape_b)
    res = F.broadcast_mul(na, nb)

    return _reduce(F, res, new_ins, keep)

def _keep_set(counter):
    return set([sub for sub, c in counter.items() if c > 0])

@_register_hybrid_op('einsum')
def _einsum(F, equation, *data, out=None, name=None):
    """
    Performing summation defined by the Eintein notation.
    """
    equation = equation.replace(' ', '')
    match = _EINSUM_PATTERN.match(equation)
    if not match:
        raise ValueError('einsum equation invalid!')
    ins1, ins2, outs = match.groups(default='')
    ins = [ins1] + [sub for sub in ins2.split(',') if sub]
    if len(ins) != len(data):
        raise ValueError(('The number of einsum inputs %d and the number of ' + \
            'input subscripts %d unequal!') % \
            (len(data), len(ins)))
    # all subscripts that appear in inputs
    in_used = set()
    for idx, (sub, idata) in enumerate(zip(ins, data)):
        if idata.ndim != len(sub):
            raise ValueError(('The numbers of axes specified in the equation ' + \
                '(%d) and in the inputs (%d) ' + \
                'do not match for input %d!') % (len(sub), idata.ndim, idx))
        in_used |= set(sub)
    out_used = set(outs)
    if len(out_used) != len(outs):
        raise ValueError('One subscript cannot appear more than once in the output!')
    if not out_used <= in_used:
        raise ValueError('Every subscript in the output must also appear in the input!')

    # reordering the axes alphabetically
    data = list(data)
    n_input = len(ins)
    for i in range(0, n_input):
        ordered_axes = list(sorted(zip(ins[i], range(0, len(ins[i]))), key=lambda x: x[0]))
        new_ins, new_axes_order = list(zip(*ordered_axes))
        new_ins = list(new_ins)
        new_data = F.transpose(data[i], axes=new_axes_order)
        cur_len = len(new_ins)
        for j in reversed(range(1, cur_len)):
            if new_ins[j] == new_ins[j - 1]:
                if new_data.shape[j] != new_data.shape[j - 1]:
                    raise ValueError('Sizes of axes with the same subscript should be equal!')
                # taking the diagonals
                new_data = F.diag(new_data, axis1=j, axis2=j - 1)
                cur_len -= 1
                if cur_len != j:
                    new_data = F.swapaxes(new_data, dim1=cur_len - 1, dim2=j - 1)
                del new_ins[j]
        data[i] = new_data
        ins[i] = new_ins

    if not outs and equation.find('->') == -1:
        outs = ''.join(sorted(in_used))

    axis_keep = Counter(outs)
    for i in range(1, n_input):
        axis_keep.update(ins[i])
    tmp_ans = data[0]
    tmp_ins = ins[0]
    tmp_ans, tmp_ins = _reduce(F, tmp_ans, tmp_ins, _keep_set(axis_keep))

    for i in range(1, n_input):
        cur_dat, cur_ins = _reduce(F, data[i], ins[i], _keep_set(axis_keep))
        axis_keep.subtract(ins[i])
        tmp_ans, tmp_ins = _contract_pair(F, tmp_ans, cur_dat, \
            tmp_ins, cur_ins, _keep_set(axis_keep))

    # tmp_ans and tmp_sub now shall contain the answer
    invert_map = dict(zip(outs, range(0, len(outs))))
    reordering = [invert_map[sub] for sub in tmp_ins]
    if not reordering:
        reordering = [0]

    return F.transpose(tmp_ans, axes=reordering, name=name, out=out)
