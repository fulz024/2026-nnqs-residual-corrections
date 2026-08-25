#pragma once

#include "FCIDUMP.hpp"

#include <cstdint>
#include <utility>
#include <vector>

class ArticleHamiltonian
{
public:
    using State = std::uint64_t;
    using MatrixElement = std::pair<State, double>;

    explicit ArticleHamiltonian(const Fcidump& fcidump);

    int n_orb() const noexcept { return n_orb_; }
    int n_elec() const noexcept { return n_elec_; }
    int ms2() const noexcept { return ms2_; }
    bool uhf() const noexcept { return uhf_; }

    double diagonal(State state) const;
    void connected(State state, std::vector<MatrixElement>& output) const;

private:
    struct SingleEntry
    {
        Orbital particle{};
        double one_body{};
        std::vector<double> two_body;
    };
    using DoubleEntry = std::pair<OrbitalPair, double>;

    int n_orb_{};
    int n_elec_{};
    int ms2_{};
    bool uhf_{};
    double core_energy_{};
    std::vector<std::vector<SingleEntry>> singles_;
    std::vector<std::vector<DoubleEntry>> doubles_;
    std::vector<double> diagonal_one_body_;
    std::vector<double> diagonal_two_body_;

    int index(Orbital i, Orbital j) const noexcept { return i * n_orb_ + j; }
    static bool occupied(State state, Orbital orbital) noexcept;
    static int parity(State state, Orbital first, Orbital second) noexcept;
    std::vector<Orbital> occupied_orbitals(State state) const;
    void construct_double_excitations(const Fcidump& fcidump);
    void construct_single_excitations(const Fcidump& fcidump);
    void construct_diagonal(const Fcidump& fcidump);
};
