#pragma once

#include <cstddef>
#include <functional>
#include <utility>

using Orbital = int;
using OrbitalPair = std::pair<Orbital, Orbital>;

struct PairHash
{
    std::size_t operator()(const OrbitalPair& value) const noexcept
    {
        const auto first = std::hash<Orbital>{}(value.first);
        const auto second = std::hash<Orbital>{}(value.second);
        return first ^ (second + 0x9e3779b9U + (first << 6U) + (first >> 2U));
    }
};
