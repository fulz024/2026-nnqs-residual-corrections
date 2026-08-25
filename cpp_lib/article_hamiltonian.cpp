#include "article_hamiltonian.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

ArticleHamiltonian::ArticleHamiltonian(const Fcidump& fcidump)
    : n_orb_(fcidump.norb),
      n_elec_(fcidump.nelec),
      ms2_(fcidump.ms2),
      uhf_(fcidump.uhf),
      core_energy_(fcidump.core_energy)
{
    if (n_orb_ < 1 || n_orb_ > 64)
    {
        throw std::invalid_argument("ArticleHamiltonian supports 1--64 spin orbitals");
    }
    construct_double_excitations(fcidump);
    construct_single_excitations(fcidump);
    construct_diagonal(fcidump);
}

bool ArticleHamiltonian::occupied(State state, Orbital orbital) noexcept
{
    return (state & (State{1} << orbital)) != 0;
}

int ArticleHamiltonian::parity(State state, Orbital first, Orbital second) noexcept
{
    if (first == second)
    {
        return 0;
    }
    const auto lo = std::min(first, second);
    const auto hi = std::max(first, second);
    const State below_hi = hi == 64 ? ~State{0} : ((State{1} << hi) - State{1});
    const State below_lo = lo == 0 ? State{0} : ((State{1} << lo) - State{1});
    return static_cast<int>(std::popcount(state & (below_hi ^ below_lo)) & 1U);
}

std::vector<Orbital> ArticleHamiltonian::occupied_orbitals(State state) const
{
    if (n_orb_ < 64)
    {
        state &= (State{1} << n_orb_) - State{1};
    }
    std::vector<Orbital> result;
    result.reserve(static_cast<std::size_t>(n_elec_));
    while (state)
    {
        const auto orbital = static_cast<Orbital>(std::countr_zero(state));
        result.push_back(orbital);
        state &= state - State{1};
    }
    return result;
}

void ArticleHamiltonian::construct_double_excitations(const Fcidump& fcidump)
{
    doubles_.resize(static_cast<std::size_t>(n_orb_ * n_orb_));
    for (const auto& [holes, row] : fcidump.two_body_integral)
    {
        const auto i = holes.first;
        const auto j = holes.second;
        if (i == j)
        {
            continue;
        }
        auto& output = doubles_.at(static_cast<std::size_t>(index(i, j)));
        for (const auto& [particles, direct] : row)
        {
            const auto a = particles.first;
            const auto b = particles.second;
            if ((i % 2) != (j % 2))
            {
                if (direct != 0.0)
                {
                    output.push_back({particles, direct});
                }
            }
            else if (a < b)
            {
                const auto antisymmetrized =
                    direct - fcidump.get_two_body_integral(i, j, b, a);
                if (antisymmetrized != 0.0)
                {
                    output.push_back({particles, antisymmetrized});
                }
            }
            else if (b < a && row.find({b, a}) == row.end())
            {
                output.push_back({{b, a}, -direct});
            }
        }
    }
}

void ArticleHamiltonian::construct_single_excitations(const Fcidump& fcidump)
{
    singles_.resize(static_cast<std::size_t>(n_orb_));
    for (Orbital i = 0; i < n_orb_; ++i)
    {
        for (Orbital a = 0; a < n_orb_; ++a)
        {
            SingleEntry entry;
            entry.particle = a;
            entry.one_body = fcidump.get_one_body_integral(i, a);
            entry.two_body.resize(static_cast<std::size_t>(n_orb_), 0.0);
            double largest = std::abs(entry.one_body);
            for (Orbital k = 0; k < n_orb_; ++k)
            {
                const double value = i <= k
                    ? fcidump.get_two_body_integral(i, k, a, k)
                        - fcidump.get_two_body_integral(i, k, k, a)
                    : fcidump.get_two_body_integral(k, i, k, a)
                        - fcidump.get_two_body_integral(k, i, a, k);
                entry.two_body[static_cast<std::size_t>(k)] = value;
                largest = std::max(largest, std::abs(value));
            }
            if (largest > 0.0)
            {
                singles_[static_cast<std::size_t>(i)].push_back(std::move(entry));
            }
        }
    }
}

void ArticleHamiltonian::construct_diagonal(const Fcidump& fcidump)
{
    diagonal_one_body_.resize(static_cast<std::size_t>(n_orb_), 0.0);
    diagonal_two_body_.resize(static_cast<std::size_t>(n_orb_ * n_orb_), 0.0);
    for (Orbital i = 0; i < n_orb_; ++i)
    {
        diagonal_one_body_[static_cast<std::size_t>(i)] =
            fcidump.get_one_body_integral(i, i);
        for (Orbital j = i + 1; j < n_orb_; ++j)
        {
            diagonal_two_body_[static_cast<std::size_t>(index(i, j))] =
                fcidump.get_two_body_integral(i, j, i, j)
                - fcidump.get_two_body_integral(i, j, j, i);
        }
    }
}

double ArticleHamiltonian::diagonal(State state) const
{
    const auto occupied_list = occupied_orbitals(state);
    double value = core_energy_;
    for (std::size_t first = 0; first < occupied_list.size(); ++first)
    {
        const auto i = occupied_list[first];
        value += diagonal_one_body_[static_cast<std::size_t>(i)];
        for (std::size_t second = first + 1; second < occupied_list.size(); ++second)
        {
            value += diagonal_two_body_[static_cast<std::size_t>(
                index(i, occupied_list[second]))];
        }
    }
    return value;
}

void ArticleHamiltonian::connected(State state, std::vector<MatrixElement>& output) const
{
    output.clear();
    const auto occupied_list = occupied_orbitals(state);

    for (const auto i : occupied_list)
    {
        for (const auto& entry : singles_[static_cast<std::size_t>(i)])
        {
            const auto a = entry.particle;
            if (occupied(state, a))
            {
                continue;
            }
            double value = entry.one_body;
            for (const auto k : occupied_list)
            {
                value += entry.two_body[static_cast<std::size_t>(k)];
            }
            if (value == 0.0)
            {
                continue;
            }
            State target = state & ~(State{1} << i);
            if (parity(target, i, a))
            {
                value = -value;
            }
            target |= State{1} << a;
            output.push_back({target, value});
        }
    }

    for (std::size_t first = 0; first < occupied_list.size(); ++first)
    {
        for (std::size_t second = first + 1; second < occupied_list.size(); ++second)
        {
            const auto i = occupied_list[first];
            const auto j = occupied_list[second];
            const auto& entries = doubles_[static_cast<std::size_t>(index(i, j))];
            for (const auto& [particles, coefficient] : entries)
            {
                const auto a = particles.first;
                const auto b = particles.second;
                if (occupied(state, a) || occupied(state, b))
                {
                    continue;
                }
                State target = state & ~(State{1} << i);
                const auto sign_first = parity(target, i, a);
                target |= State{1} << a;
                target &= ~(State{1} << j);
                const auto sign_second = parity(target, j, b);
                target |= State{1} << b;
                output.push_back({target, (sign_first ^ sign_second) ? -coefficient : coefficient});
            }
        }
    }
}
