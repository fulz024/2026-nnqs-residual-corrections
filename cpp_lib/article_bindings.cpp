#include "FCIDUMP.hpp"
#include "article_hamiltonian.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace py = pybind11;

namespace
{
template <typename T>
py::array_t<T> numpy_copy(const std::vector<T>& values)
{
    py::array_t<T> output(values.size());
    if (!values.empty())
    {
        std::memcpy(output.mutable_data(), values.data(), values.size() * sizeof(T));
    }
    return output;
}

py::dict generate_couplings(
    const ArticleHamiltonian& hamiltonian,
    const py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>& states)
{
    if (states.ndim() != 1)
    {
        throw py::value_error("source_states must be a one-dimensional uint64 array");
    }
    const auto input = states.unchecked<1>();
    std::vector<std::uint64_t> source;
    std::vector<std::uint64_t> target;
    std::vector<std::int64_t> lengths;
    std::vector<double> coefficients;
    std::vector<double> target_diagonal;
    source.reserve(static_cast<std::size_t>(input.shape(0)));
    lengths.reserve(static_cast<std::size_t>(input.shape(0)));

    std::vector<ArticleHamiltonian::MatrixElement> connected;
    for (py::ssize_t row = 0; row < input.shape(0); ++row)
    {
        const auto state = input(row);
        source.push_back(state);
        std::int64_t length = 0;

        target.push_back(state);
        coefficients.push_back(hamiltonian.diagonal(state));
        target_diagonal.push_back(coefficients.back());
        ++length;

        hamiltonian.connected(state, connected);
        for (const auto& [coupled_state, coefficient] : connected)
        {
            target.push_back(coupled_state);
            coefficients.push_back(coefficient);
            target_diagonal.push_back(hamiltonian.diagonal(coupled_state));
            ++length;
        }
        lengths.push_back(length);
    }

    py::dict output;
    output["source_states"] = numpy_copy(source);
    output["coupled_states"] = numpy_copy(target);
    output["coupled_states_length"] = numpy_copy(lengths);
    output["coefficients"] = numpy_copy(coefficients);
    output["coupled_diagonal"] = numpy_copy(target_diagonal);
    return output;
}

py::array_t<double> diagonal(
    const ArticleHamiltonian& hamiltonian,
    const py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>& states)
{
    if (states.ndim() != 1)
    {
        throw py::value_error("states must be a one-dimensional uint64 array");
    }
    const auto input = states.unchecked<1>();
    py::array_t<double> output(input.shape(0));
    auto result = output.mutable_unchecked<1>();
    for (py::ssize_t index = 0; index < input.shape(0); ++index)
    {
        result(index) = hamiltonian.diagonal(input(index));
    }
    return output;
}
}  // namespace

PYBIND11_MODULE(residual_coupling_module, module)
{
    module.doc() = "Article-only determinant couplings for restricted energy and residual PT2";

    py::class_<Fcidump>(module, "Fcidump")
        .def(py::init<const std::string&>())
        .def_readonly("norb", &Fcidump::norb)
        .def_readonly("nelec", &Fcidump::nelec)
        .def_readonly("ms2", &Fcidump::ms2)
        .def_readonly("uhf", &Fcidump::uhf);

    py::class_<ArticleHamiltonian>(module, "Hamiltonian")
        .def(py::init<const Fcidump&>())
        .def_property_readonly("norb", &ArticleHamiltonian::n_orb)
        .def_property_readonly("nelec", &ArticleHamiltonian::n_elec)
        .def_property_readonly("ms2", &ArticleHamiltonian::ms2)
        .def_property_readonly("uhf", &ArticleHamiltonian::uhf);

    module.def(
        "generate_couplings",
        &generate_couplings,
        py::arg("hamiltonian"),
        py::arg("source_states"));
    module.def("diagonal", &diagonal, py::arg("hamiltonian"), py::arg("states"));
}
