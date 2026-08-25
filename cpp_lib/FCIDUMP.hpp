#pragma once

#include "article_types.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

class Fcidump
{
public:
    using value_type = double;
    using TwoBodyRow = std::unordered_map<OrbitalPair, value_type, PairHash>;

    explicit Fcidump(const std::string& file_path);

    int norb{};  // Spin orbitals. RHF input is expanded to alpha/beta orbitals.
    int nelec{};
    int ms2{};
    bool uhf{};
    value_type core_energy{};
    std::unordered_map<OrbitalPair, TwoBodyRow, PairHash> two_body_integral;
    std::vector<std::vector<value_type>> one_body_integral;

    value_type get_two_body_integral(Orbital i, Orbital j, Orbital a, Orbital b) const;
    value_type get_one_body_integral(Orbital i, Orbital a) const;

private:
    void from_file(const std::string& file_path);
    void read_integral_uhf(std::ifstream& stream);
    void read_integral_rhf(std::ifstream& stream);
    void insert_two_body(Orbital i, Orbital j, Orbital a, Orbital b, value_type value);
    void insert_rhf_two_body(Orbital i, Orbital j, Orbital a, Orbital b, value_type value);

    static Orbital alpha(Orbital spatial) { return 2 * spatial - 2; }
    static Orbital beta(Orbital spatial) { return 2 * spatial - 1; }

    template <typename T>
    static T read_parameter(
        const std::string& header,
        const std::string& name,
        const std::string& expression)
    {
        std::regex pattern(name + expression, std::regex::icase);
        std::smatch match;
        if (!std::regex_search(header, match, pattern))
        {
            throw std::invalid_argument(name + " is not present in the FCIDUMP header");
        }
        std::string raw = match[1];
        T value{};
        if constexpr (std::is_same_v<T, bool>)
        {
            std::transform(raw.begin(), raw.end(), raw.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            std::istringstream(raw) >> std::boolalpha >> value;
        }
        else
        {
            std::istringstream(raw) >> value;
        }
        return value;
    }
};
