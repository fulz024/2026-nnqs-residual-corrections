#include "FCIDUMP.hpp"

Fcidump::Fcidump(const std::string& file_path)
{
    from_file(file_path);
}

void Fcidump::from_file(const std::string& file_path)
{
    try
    {
        std::ifstream stream(file_path);
        if (!stream)
        {
            throw std::invalid_argument("failed to open the file");
        }

        std::string header;
        std::string line;
        bool ended = false;
        while (std::getline(stream, line))
        {
            std::string upper = line;
            std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) {
                return static_cast<char>(std::toupper(c));
            });
            const auto end_position = upper.find("&END");
            const auto slash_position = line.find('/');
            const auto stop = std::min(
                end_position == std::string::npos ? line.size() : end_position,
                slash_position == std::string::npos ? line.size() : slash_position);
            header += " " + line.substr(0, stop);
            if (end_position != std::string::npos || slash_position != std::string::npos)
            {
                ended = true;
                break;
            }
        }
        if (!ended)
        {
            throw std::invalid_argument("unterminated FCIDUMP header");
        }

        const std::string integer = R"([ ]*=[ ]*([-+]?\d+))";
        const std::string boolean = R"([ ]*=[ .]*(FALSE|TRUE))";
        norb = read_parameter<int>(header, "NORB", integer);
        nelec = read_parameter<int>(header, "NELEC", integer);
        ms2 = read_parameter<int>(header, "MS2", integer);
        try
        {
            uhf = read_parameter<bool>(header, "UHF", boolean);
        }
        catch (const std::invalid_argument&)
        {
            uhf = false;
        }
        if (!uhf)
        {
            norb *= 2;
        }
        if (norb < 1 || norb > 64)
        {
            throw std::invalid_argument("the public article evaluator supports 1--64 spin orbitals");
        }

        one_body_integral.assign(
            static_cast<std::size_t>(norb),
            std::vector<value_type>(static_cast<std::size_t>(norb), 0.0));
        if (uhf)
        {
            read_integral_uhf(stream);
        }
        else
        {
            read_integral_rhf(stream);
        }
    }
    catch (const std::exception& error)
    {
        throw std::invalid_argument(
            "cannot read FCIDUMP file " + file_path + ": " + error.what());
    }
}

void Fcidump::read_integral_uhf(std::ifstream& stream)
{
    Orbital i{}, a{}, j{}, b{};
    value_type value{};
    while (stream >> value >> i >> a >> j >> b)
    {
        if (i && a && j && b)
        {
            insert_two_body(i - 1, j - 1, a - 1, b - 1, value);
            insert_two_body(i - 1, b - 1, a - 1, j - 1, value);
            insert_two_body(a - 1, j - 1, i - 1, b - 1, value);
            insert_two_body(a - 1, b - 1, i - 1, j - 1, value);
        }
        else if (a)
        {
            one_body_integral.at(i - 1).at(a - 1) = value;
            one_body_integral.at(a - 1).at(i - 1) = value;
        }
        else if (!i)
        {
            core_energy = value;
        }
    }
}

void Fcidump::read_integral_rhf(std::ifstream& stream)
{
    Orbital i{}, a{}, j{}, b{};
    value_type value{};
    while (stream >> value >> i >> a >> j >> b)
    {
        if (i && a && j && b)
        {
            insert_rhf_two_body(i, j, a, b, value);
            insert_rhf_two_body(i, b, a, j, value);
            insert_rhf_two_body(a, j, i, b, value);
            insert_rhf_two_body(a, b, i, j, value);
        }
        else if (a)
        {
            one_body_integral.at(alpha(i)).at(alpha(a)) = value;
            one_body_integral.at(beta(i)).at(beta(a)) = value;
            one_body_integral.at(alpha(a)).at(alpha(i)) = value;
            one_body_integral.at(beta(a)).at(beta(i)) = value;
        }
        else if (!i)
        {
            core_energy = value;
        }
    }
}

void Fcidump::insert_rhf_two_body(
    Orbital i,
    Orbital j,
    Orbital a,
    Orbital b,
    value_type value)
{
    insert_two_body(alpha(i), alpha(j), alpha(a), alpha(b), value);
    insert_two_body(beta(i), beta(j), beta(a), beta(b), value);
    insert_two_body(alpha(i), beta(j), alpha(a), beta(b), value);
    insert_two_body(beta(i), alpha(j), beta(a), alpha(b), value);
}

void Fcidump::insert_two_body(
    Orbital i,
    Orbital j,
    Orbital a,
    Orbital b,
    value_type value)
{
    if (i <= j)
    {
        two_body_integral[{i, j}].insert({{a, b}, value});
    }
    if (i >= j)
    {
        two_body_integral[{j, i}].insert({{b, a}, value});
    }
}

Fcidump::value_type Fcidump::get_two_body_integral(
    Orbital i,
    Orbital j,
    Orbital a,
    Orbital b) const
{
    const auto row = two_body_integral.find({i, j});
    if (row == two_body_integral.end())
    {
        return 0.0;
    }
    const auto entry = row->second.find({a, b});
    return entry == row->second.end() ? 0.0 : entry->second;
}

Fcidump::value_type Fcidump::get_one_body_integral(Orbital i, Orbital a) const
{
    return one_body_integral.at(static_cast<std::size_t>(i)).at(static_cast<std::size_t>(a));
}
